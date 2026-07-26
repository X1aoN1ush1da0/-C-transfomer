#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#if defined(_M_AVX2)
#include <immintrin.h>
#endif
#ifdef _DEBUG
#include <crtdbg.h>
#endif

struct tensor {
	float* value;
	int row;
	int col;
	int stride0;
	int stride1;
	int value_owner;

	float* grad;
	int requires_grad;
	int parent_cnt;
	struct tensor** parents;
	void(*backward_fn)(struct tensor* self);
	void* ctx;
	void(*free_ctx)(void* ctx);
	int mark;
};

struct layer;

struct model {
	tensor* src_E;
	tensor* tgt_E;
	tensor* W_project;
	layer* layers;
	tensor* params[600];
	float* adam_m[600];
	float* adam_v[600];
	int param_cnt;
	int adam_step;
};

struct scale_ctx {
	float b;
};

struct embedding_ctx {
	int n;
	int* list;
};

struct cross_entropy_loss_ctx {
	int* targets;
	tensor* ex;
};

void free_tensor(tensor* t);


void free_embedding_ctx(void* p) {
	embedding_ctx* ctx = (embedding_ctx*)p;
	free(ctx->list);
	free(ctx);
}

void free_cross_entropy_loss_ctx(void* p) {
	cross_entropy_loss_ctx* ctx = (cross_entropy_loss_ctx*)p;
	free_tensor(ctx->ex);
	free(ctx);
}
const int H = 6;
const float sipu = 0.000001f;
const int dk = 24;
const int dm = H * dk;
const int dff = 4 * dm;
const int N = 6;

uint32_t train_rng_state = 2463534242u;

uint32_t next_train_random() {
	train_rng_state ^= train_rng_state << 13;
	train_rng_state ^= train_rng_state >> 17;
	train_rng_state ^= train_rng_state << 5;
	return train_rng_state;
}

tensor* make_tensor(int row, int col, int stride0, int stride1) {
	tensor* t = (tensor*)malloc(sizeof(tensor));
	t->row = row;
	t->col = col;
	t->stride0 = stride0;
	t->stride1 = stride1;
	t->value_owner = 1;
	int size = (row - 1) * stride0 + (col - 1) * stride1 + 1;
	t->value = (float*)malloc(size * sizeof(float));

	t->grad = NULL;
	t->requires_grad = 0;
	t->parent_cnt = 0;
	t->parents = NULL;
	t->backward_fn = NULL;
	t->ctx = NULL;
	t->free_ctx = NULL;
	t->mark = 0;
	return t;
}

tensor* make_tensor_view(float* value, int row, int col, int stride0, int stride1) {
	tensor* t = (tensor*)malloc(sizeof(tensor));
	t->value = value;
	t->row = row;
	t->col = col;
	t->stride0 = stride0;
	t->stride1 = stride1;
	t->value_owner = 0;

	t->grad = NULL;
	t->requires_grad = 0;
	t->parent_cnt = 0;
	t->parents = NULL;
	t->backward_fn = NULL;
	t->ctx = NULL;
	t->free_ctx = NULL;
	t->mark = 0;
	return t;
}

void free_tensor(tensor* t) {
	if (t->value_owner) {
		free(t->value);
	}
	free(t->grad);
	free(t->parents);
	if (t->ctx) {
		if (t->free_ctx) {
			t->free_ctx(t->ctx);
		}
		else {
			free(t->ctx);
		}
	}
	free(t);
}

float at(tensor* a, int i, int j) {
	return a->value[i * a->stride0 + j * a->stride1];
}

void set(tensor* a, int i, int j, float x) {
	a->value[i * a->stride0 + j * a->stride1] = x;
}

int storage_size(tensor* t) {
	return (t->row - 1) * t->stride0 + (t->col - 1) * t->stride1 + 1;
}

float& grad_at(tensor* t, int i, int j) {
	return t->grad[i * t->stride0 + j * t->stride1];
}


void ensure_grad(tensor* t) {
	if (t->grad) return;
	int size = storage_size(t);
	t->grad = (float*)calloc(size, sizeof(float));
}

void add_grad(tensor* tgt, tensor* src) {
	if (!tgt->requires_grad) return;
	ensure_grad(tgt);
	for (int i = 0; i < tgt->row; i++) {
		for (int j = 0; j < tgt->col; j++) {
			grad_at(tgt, i, j) += grad_at(src, i, j);
		}
	}
}


void add_backward(tensor* self) {
	tensor* a = self->parents[0];
	tensor* b = self->parents[1];
	if (a->requires_grad) {
		ensure_grad(a);
		for (int i = 0; i < self->row; i++) {
			for (int j = 0; j < self->col; j++) {
				grad_at(a, i, j) += grad_at(self, i, j);
			}
		}
	}
	if (b->requires_grad) {
		ensure_grad(b);
		for (int i = 0; i < self->row; i++) {
			for (int j = 0; j < self->col; j++) {
				grad_at(b, i, j) += grad_at(self, i, j);
			}
		}
	}
}

#if defined(_M_AVX2)
float sum8(__m256 x) {
	__m128 low = _mm256_castps256_ps128(x);
	__m128 high = _mm256_extractf128_ps(x, 1);
	__m128 sum = _mm_add_ps(low, high);
	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);
	return _mm_cvtss_f32(sum);
}
#endif

void mul_backward(tensor* self) {
	tensor* a = self->parents[0];
	tensor* b = self->parents[1];
	if (a->requires_grad) {
		ensure_grad(a);
		if (a->stride1 == 1 && b->stride1 == 1 && self->stride1 == 1) {
			for (int i = 0; i < self->row; i++) {
				float* a_grad_row = a->grad + i * a->stride0;
				const float* out_grad_row = self->grad + i * self->stride0;
				for (int j = 0; j < b->row; j++) {
					const float* b_row = b->value + j * b->stride0;
					float sum = 0;
					int k = 0;
#if defined(_M_AVX2)
					__m256 sum8_now = _mm256_setzero_ps();
					for (; k + 7 < self->col; k += 8) {
						__m256 dout8 = _mm256_loadu_ps(out_grad_row + k);
						__m256 b8 = _mm256_loadu_ps(b_row + k);
						sum8_now = _mm256_add_ps(sum8_now, _mm256_mul_ps(dout8, b8));
					}
					sum = sum8(sum8_now);
#endif
					for (; k < self->col; k++) {
						sum += out_grad_row[k] * b_row[k];
					}
					a_grad_row[j] += sum;
				}
			}
		}
		else {
		for (int i = 0; i < self->row; i++) {
			for (int j = 0; j < b->row; j++) {
				float sum = 0;
				for (int k = 0; k < self->col; k++) {
					sum += grad_at(self, i, k) * at(b, j, k);
				}
				grad_at(a, i, j) += sum;
			}
		}
		}
	}
	if (b->requires_grad) {
		ensure_grad(b);
		if (a->stride1 == 1 && b->stride1 == 1 && self->stride1 == 1) {
			for (int k = 0; k < a->col; k++) {
				float* b_grad_row = b->grad + k * b->stride0;
				for (int i = 0; i < self->row; i++) {
					float av = a->value[i * a->stride0 + k];
					const float* out_grad_row = self->grad + i * self->stride0;
					int j = 0;
#if defined(_M_AVX2)
					__m256 a8 = _mm256_set1_ps(av);
					for (; j + 7 < self->col; j += 8) {
						__m256 grad8 = _mm256_loadu_ps(b_grad_row + j);
						__m256 dout8 = _mm256_loadu_ps(out_grad_row + j);
						_mm256_storeu_ps(b_grad_row + j, _mm256_add_ps(grad8, _mm256_mul_ps(a8, dout8)));
					}
#endif
					for (; j < self->col; j++) {
						b_grad_row[j] += av * out_grad_row[j];
					}
				}
			}
		}
		else {
		for (int i = 0; i < a->col; i++) {
			for (int j = 0; j < self->col; j++) {
				float sum = 0;
				for (int k = 0; k < self->row; k++) {
					sum += at(a, k, i) * grad_at(self, k, j);
				}
				grad_at(b, i, j) += sum;
			}
		}
		}
	}
}

void scale_backward(tensor* self) {
	tensor* a = self->parents[0];
	scale_ctx* ctx = (scale_ctx*)self->ctx;
	if (a->requires_grad) {
		ensure_grad(a);
		for (int i = 0; i < self->row; i++) {
			for (int j = 0; j < self->col; j++) {
				grad_at(a, i, j) += grad_at(self, i, j) * ctx->b;
			}
		}
	}
}

void transpose_backward(tensor* self) {
	tensor* a = self->parents[0];
	if (!a->requires_grad) return;
	ensure_grad(a);
	for (int i = 0; i < self->row; i++) {
		for (int j = 0; j < self->col; j++) {
			grad_at(a, j, i) += grad_at(self, i, j);
		}
	}
}

void relu_backward(tensor* self) {
	tensor* a = self->parents[0];
	if (!a->requires_grad) return;
	ensure_grad(a);
	for (int i = 0; i < self->row; i++) {
		for (int j = 0; j < self->col; j++) {
			if (at(a, i, j) > 0)
				grad_at(a, i, j) += grad_at(self, i, j);
		}
	}
}

void soft_max_backward(tensor* self) {
	tensor* a = self->parents[0];
	if (!a->requires_grad) return;
	ensure_grad(a);
	for (int i = 0; i < self->row; i++) {
		float f = 0;
		for (int j = 0; j < self->col; j++) {
			f += grad_at(self, i, j) * at(self, i, j);
		}
		for (int j = 0; j < self->col; j++) {
			grad_at(a, i, j) += at(self, i, j) * (grad_at(self, i, j) - f);
		}
	}
}

void cat_backward(tensor* self) {
	for (int h = 0; h < H; h++) {
		tensor* a = self->parents[h];
		if (!a->requires_grad) continue;
		ensure_grad(a);
		for (int i = 0; i < self->row; i++) {
			for (int j = 0; j < dk; j++) {
				grad_at(a, i, j) += grad_at(self, i, h * dk + j);
			}
		}
	}
}

void embedding_backward(tensor* self) {
	tensor* E = self->parents[0];
	embedding_ctx* ctx = (embedding_ctx*)self->ctx;
	if (!E->requires_grad) return;
	ensure_grad(E);
	for (int i = 0; i < ctx->n; i++) {
		int id = ctx->list[i];
		for (int j = 0; j < self->col; j++) {
			grad_at(E, id, j) += grad_at(self, i, j);
		}
	}
}

void layernorm_backward(tensor* self) {
	int n = self->row;
	int d = self->col;
	tensor* a = self->parents[0];
	tensor* gamma = self->parents[1];
	tensor* beta = self->parents[2];
	if (a->requires_grad) ensure_grad(a);
	if (gamma->requires_grad) ensure_grad(gamma);
	if (beta->requires_grad) ensure_grad(beta);
	for (int i = 0; i < n; i++) {
		float mean = 0;
		for (int j = 0; j < d; j++) {
			mean += at(a, i, j);
		}
		mean /= d;
		float var = 0;
		for (int j = 0; j < d; j++) {
			float v = at(a, i, j) - mean;
			var += v * v;
		}
		var /= d;
		float inv = 1.0f / sqrtf(var + sipu);

		for (int j = 0; j < d; j++) {
			if (gamma->requires_grad)
				grad_at(gamma, 0, j) += grad_at(self, i, j) * (at(a, i, j) - mean) * inv;
			if (beta->requires_grad)
				grad_at(beta, 0, j) += grad_at(self, i, j);
		}

		float sum1 = 0;
		float sum2 = 0;
		for (int j = 0; j < d; j++) {
			float tmp1 = grad_at(self, i, j);
			float tmp2 = gamma->value[j];
			sum1 += tmp1 * tmp2;
			sum2 += tmp1 * tmp2 * ((at(a, i, j) - mean) * inv);
		}

		for (int j = 0; j < d; j++) {
			if (a->requires_grad) {
				float now = grad_at(self, i, j) * gamma->value[j];
				grad_at(a, i, j) += inv * (now - sum1 / d - (at(a, i, j) - mean) * inv * sum2 / d);
			}
		}
	}
}

void cross_entropy_loss_backward(tensor* self) {
	tensor* logits = self->parents[0];
	cross_entropy_loss_ctx* ctx = (cross_entropy_loss_ctx*)(self->ctx);
	int* targets = ctx->targets;
	tensor* ex = ctx->ex;
	ensure_grad(logits);
	for (int i = 0; i < ex->row; i++) {
		float sum = 0;
		for (int j = 0; j < ex->col; j++) {
			sum += at(ex, i, j);
		}
		for (int j = 0; j < ex->col; j++) {
			if (j == targets[i]) {
				grad_at(logits, i, j) += grad_at(self, 0, 0) * (at(ex, i, j) / sum - 1) / ex->row;
			}
			else {
				grad_at(logits, i, j) += grad_at(self, 0, 0) * (at(ex, i, j) / sum) / ex->row;
			}
		}
	}
}




void copy(tensor* a, tensor* b) {
	int size = a->row * a->col;
	for (int i = 0; i < size; i++) {
		b->value[i] = a->value[i];
	}
}

void mul(tensor* c, tensor* a, tensor* b) {
	int p = a->row;
	int q = a->col;
	int r = b->col;
	// Most projections use contiguous X, W, and output rows. Keep the stride path below for views.
	if (a->stride1 == 1 && b->stride1 == 1 && c->stride1 == 1) {
		for (int i = 0; i < p; i++) {
			float* c_row = c->value + i * c->stride0;
			const float* a_row = a->value + i * a->stride0;
			for (int j = 0; j < r; j++) {
				c_row[j] = 0;
			}
			for (int k = 0; k < q; k++) {
				const float* b_row = b->value + k * b->stride0;
				float av = a_row[k];
				int j = 0;
#if defined(_M_AVX2)
				__m256 a8 = _mm256_set1_ps(av);
				for (; j + 7 < r; j += 8) {
					__m256 c8 = _mm256_loadu_ps(c_row + j);
					__m256 b8 = _mm256_loadu_ps(b_row + j);
					_mm256_storeu_ps(c_row + j, _mm256_add_ps(c8, _mm256_mul_ps(a8, b8)));
				}
#endif
				for (; j < r; j++) {
					c_row[j] += av * b_row[j];
				}
			}
		}
	}
	else {
	for (int i = 0; i < p; i++) {
		for (int j = 0; j < r; j++) {
			float tmp = 0;
			for (int k = 0; k < q; k++) {
				tmp += at(a, i, k) * at(b, k, j);
			}
			set(c, i, j, tmp);
		}
	}
	}

	c->requires_grad = a->requires_grad || b->requires_grad;
	if (!c->requires_grad) return;

	c->parent_cnt = 2;
	c->parents = (tensor**)malloc(2 * sizeof(tensor*));
	c->parents[0] = a;
	c->parents[1] = b;
	c->backward_fn = mul_backward;
}

void add(tensor* c, tensor* a, tensor* b) {
	int size = a->row * a->col;
	for (int i = 0; i < size; i++) {
		c->value[i] = a->value[i] + b->value[i];
	}

	c->requires_grad = a->requires_grad || b->requires_grad;
	if (!c->requires_grad) return;

	c->parent_cnt = 2;
	c->parents = (tensor**)malloc(2 * sizeof(tensor*));
	c->parents[0] = a;
	c->parents[1] = b;
	c->backward_fn = add_backward;
}

/*void hul(tensor* c, tensor* a, tensor* b) {
	int size = a->row * a->col;
	for (int i = 0; i < size; i++) {
		c->value[i] = a->value[i] * b->value[i];
	}

	c->requires_grad = a->requires_grad || b->requires_grad;
	if (!c->requires_grad) return;

	c->parent_cnt = 2;
	c->parents = (tensor**)malloc(2 * sizeof(tensor*));
	c->parents[0] = a;
	c->parents[1] = b;
	c->backward_fn = hul_backward;
}*/

/*void sub(tensor* c, tensor* a, tensor* b) {
	int size = a->row * a->col;
	for (int i = 0; i < size; i++) {
		c->value[i] = a->value[i] - b->value[i];
	}

	c->requires_grad = a->requires_grad || b->requires_grad;
	if (!c->requires_grad) return;

	c->parent_cnt = 2;
	c->parents = (tensor**)malloc(2 * sizeof(tensor*));
	c->parents[0] = a;
	c->parents[1] = b;
	c->backward_fn = sub_backward;
}*/

void scale_tensor(tensor* out, tensor* a, float b) {
	int size = a->row * a->col;
	for (int i = 0; i < size; i++) {
		out->value[i] = a->value[i] * b;
	}

	out->requires_grad = a->requires_grad;
	if (!out->requires_grad) return;

	out->parent_cnt = 1;
	out->parents = (tensor**)malloc(sizeof(tensor*));
	out->parents[0] = a;
	scale_ctx* ctx = (scale_ctx*)malloc(sizeof(scale_ctx));
	ctx->b = b;
	out->ctx = ctx;
	out->backward_fn = scale_backward;
}

/*float sum(tensor* a) {
	float s = 0;
	int size = a->row * a->col;
	for (int i = 0; i < size; i++) {
		s += a->value[i];
	}
	return s;
}*/

tensor* transpose(tensor* a) {
	tensor* out = make_tensor_view(a->value, a->col, a->row, a->stride1, a->stride0);
	out->requires_grad = a->requires_grad;
	if (!out->requires_grad) return out;

	out->parent_cnt = 1;
	out->parents = (tensor**)malloc(sizeof(tensor*));
	out->parents[0] = a;
	out->backward_fn = transpose_backward;
	return out;
}

/*float max(tensor* t) {
	float max_now = t->value[0];
	int size = t->row * t->col;
	for (int i = 1; i < size; i++) {
		if (t->value[i] > max_now) {
			max_now = t->value[i];
		}
	}
	return max_now;
}*/

/*void exp_tensor(tensor* b, tensor* a) {
	int size = a->row * a->col;
	for (int i = 0; i < size; i++) {
		b->value[i] = expf(a->value[i]);
	}
}*/

/*void sum_row(tensor* b, tensor* a) {
	for (int i = 0; i < a->row; i++) {
		float s = 0;
		for (int j = 0; j < a->col; j++) {
			s += at(a, i, j);
		}
		set(b, i, 0, s);
	}
}*/

/*void sum_col(tensor* b, tensor* a) {
	for (int j = 0; j < a->col; j++) {
		float s = 0;
		for (int i = 0; i < a->row; i++) {
			s += at(a, i, j);
		}
		set(b, 0, j, s);
	}
}*/

/*void max_row(tensor* b, tensor* a) {
	for (int i = 0; i < a->row; i++) {
		float max_now = at(a, i, 0);
		for (int j = 1; j < a->col; j++) {
			float s = at(a, i, j);
			if (max_now < s) {
				max_now = s;
			}
		}
		set(b, i, 0, max_now);
	}
}*/

//行softmax算子
void soft_max(tensor* b, tensor* a) {
	for (int i = 0; i < a->row; i++) {
		float max_now = at(a, i, 0);
		for (int j = 1; j < a->col; j++) {
			float x = at(a, i, j);
			if (x > max_now) {
				max_now = x;
			}
		}
		float row_sum = 0;
		for (int j = 0; j < a->col; j++) {
			float x = expf(at(a, i, j) - max_now);
			set(b, i, j, x);
			row_sum += x;
		}
		for (int j = 0; j < a->col; j++) {
			set(b, i, j, at(b, i, j) / row_sum);
		}
	}

	b->requires_grad = a->requires_grad;
	if (!b->requires_grad) return;

	b->parent_cnt = 1;
	b->parents = (tensor**)malloc(sizeof(tensor*));
	b->parents[0] = a;
	b->backward_fn = soft_max_backward;
}

//relu算子
void relu_tensor(tensor* out, tensor* a) {
	int size = a->row * a->col;
	for (int i = 0; i < size; i++) {
		if (a->value[i] < 0) {
			out->value[i] = 0;
			continue;
		}
		out->value[i] = a->value[i];
	}

	out->requires_grad = a->requires_grad;
	if (!out->requires_grad) return;

	out->parent_cnt = 1;
	out->parents = (tensor**)malloc(sizeof(tensor*));
	out->parents[0] = a;
	out->backward_fn = relu_backward;
}

void cat_heads(tensor* out, tensor** head_out) {
	int n = out->row;
	for (int h = 0; h < H; h++) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < dk; j++) {
				out->value[i * out->stride0 + h * dk + j] = head_out[h]->value[i * head_out[h]->stride0 + j];
			}
		}
	}

	out->requires_grad = 0;
	for (int h = 0; h < H; h++) {
		if (head_out[h]->requires_grad) {
			out->requires_grad = 1;
			break;
		}
	}
	if (!out->requires_grad) return;

	out->parent_cnt = H;
	out->parents = (tensor**)malloc(H * sizeof(tensor*));
	for (int h = 0; h < H; h++) {
		out->parents[h] = head_out[h];
	}
	out->backward_fn = cat_backward;
}



//单层类
struct encoder_layer {
	tensor* Wq[H];
	tensor* Wk[H];
	tensor* Wv[H];
	tensor* Wo;
	tensor* gamma[2];
	tensor* beta[2];
	tensor* W1;
	tensor* W2;
};

struct decoder_layer {
	tensor* Wq[H];
	tensor* Wk[H];
	tensor* Wv[H];
	tensor* Wqc[H];
	tensor* Wkc[H];
	tensor* Wvc[H];
	tensor* Wo;
	tensor* Wo_cross;
	tensor* gamma[3];
	tensor* beta[3];
	tensor* W1;
	tensor* W2;
};

//多层管理
struct layer {
	encoder_layer* layerlist[N];
	decoder_layer* de_layerlist[N];
};


void add_param(model* m, tensor* t) {
	t->requires_grad = 1;
	int index = m->param_cnt;
	m->params[index] = t;
	int size = storage_size(t);
	m->adam_m[index] = (float*)calloc(size, sizeof(float));
	m->adam_v[index] = (float*)calloc(size, sizeof(float));
	m->param_cnt += 1;
}

//张量随机
void random_tensor(tensor* t) {
	float limit = sqrtf(6.0f / (t->row + t->col));
	for (int i = 0; i < t->row; i++) {
		for (int j = 0; j < t->col; j++) {
			float x = (float)rand() / RAND_MAX;
			set(t, i, j, (x * 2.0f - 1.0f) * limit);
		}
	}
}

//随机权重
tensor* make_param(model* m, int row, int col) {
	tensor* t = make_tensor(row, col, col, 1);
	random_tensor(t);
	add_param(m, t);
	return t;
}

//特定权重
tensor* make_param_fill(model* m, int row, int col, float value) {
	tensor* t = make_tensor(row, col, col, 1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			set(t, i, j, value);
		}
	}
	add_param(m, t);
	return t;
}

//编码器随机权重
encoder_layer* make_encoder_layer(model* m) {
	encoder_layer* a = (encoder_layer*)malloc(sizeof(encoder_layer));
	for (int h = 0; h < H; h++) {
		a->Wq[h] = make_param(m, dm, dk);
		a->Wk[h] = make_param(m, dm, dk);
		a->Wv[h] = make_param(m, dm, dk);
	}
	a->Wo = make_param(m, dm, dm);
	for (int i = 0; i < 2; i++) {
		a->gamma[i] = make_param_fill(m, 1, dm, 1.0f);
		a->beta[i] = make_param_fill(m, 1, dm, 0.0f);
	}
	a->W1 = make_param(m, dm, dff);
	a->W2 = make_param(m, dff, dm);
	return a;
}

//解码器随机权重
decoder_layer* make_decoder_layer(model* m) {
	decoder_layer* a = (decoder_layer*)malloc(sizeof(decoder_layer));
	for (int h = 0; h < H; h++) {
		a->Wq[h] = make_param(m, dm, dk);
		a->Wk[h] = make_param(m, dm, dk);
		a->Wv[h] = make_param(m, dm, dk);
		a->Wqc[h] = make_param(m, dm, dk);
		a->Wkc[h] = make_param(m, dm, dk);
		a->Wvc[h] = make_param(m, dm, dk);
	}
	a->Wo = make_param(m, dm, dm);
	a->Wo_cross = make_param(m, dm, dm);
	for (int i = 0; i < 3; i++) {
		a->gamma[i] = make_param_fill(m, 1, dm, 1.0f);
		a->beta[i] = make_param_fill(m, 1, dm, 0.0f);
	}
	a->W1 = make_param(m, dm, dff);
	a->W2 = make_param(m, dff, dm);
	return a;
}

//全模型权重初始化
model* make_model(int src_vocab, int tgt_vocab) {
	model* m = (model*)malloc(sizeof(model));
	m->param_cnt = 0;
	m->adam_step = 0;
	m->src_E = make_param(m, src_vocab, dm);
	m->tgt_E = make_param(m, tgt_vocab, dm);
	m->W_project = make_param(m, dm, tgt_vocab);
	m->layers = (layer*)malloc(sizeof(layer));
	for (int i = 0; i < N; i++) {
		m->layers->layerlist[i] = make_encoder_layer(m);
		m->layers->de_layerlist[i] = make_decoder_layer(m);
	}
	return m;
}

void free_model(model* m) {
	for (int i = 0; i < m->param_cnt; i++) {
		free(m->adam_m[i]);
		free(m->adam_v[i]);
		free_tensor(m->params[i]);
	}
	for (int i = 0; i < N; i++) {
		free(m->layers->layerlist[i]);
		free(m->layers->de_layerlist[i]);
	}
	free(m->layers);
	free(m);
}

void save_model(model* m, const char* path) {
	FILE* f = NULL;
	if (fopen_s(&f, path, "wb") != 0) return;
	int src_vocab = m->src_E->row;
	int tgt_vocab = m->tgt_E->row;
	fwrite(&src_vocab, sizeof(int), 1, f);
	fwrite(&tgt_vocab, sizeof(int), 1, f);
	fwrite(&m->param_cnt, sizeof(int), 1, f);
	for (int i = 0; i < m->param_cnt; i++) {
		tensor* t = m->params[i];
		fwrite(&t->row, sizeof(int), 1, f);
		fwrite(&t->col, sizeof(int), 1, f);
		for (int r = 0; r < t->row; r++) {
			for (int c = 0; c < t->col; c++) {
				float x = at(t, r, c);
				fwrite(&x, sizeof(float), 1, f);
			}
		}
	}
	fwrite(&m->adam_step, sizeof(int), 1, f);
	for (int i = 0; i < m->param_cnt; i++) {
		int size = storage_size(m->params[i]);
		fwrite(m->adam_m[i], sizeof(float), size, f);
		fwrite(m->adam_v[i], sizeof(float), size, f);
	}
	fclose(f);
}

model* load_model(const char* path) {
	FILE* f = NULL;
	if (fopen_s(&f, path, "rb") != 0) return NULL;
	int src_vocab;
	int tgt_vocab;
	int param_cnt;
	fread(&src_vocab, sizeof(int), 1, f);
	fread(&tgt_vocab, sizeof(int), 1, f);
	fread(&param_cnt, sizeof(int), 1, f);
	model* m = make_model(src_vocab, tgt_vocab);
	for (int i = 0; i < param_cnt; i++) {
		int row;
		int col;
		fread(&row, sizeof(int), 1, f);
		fread(&col, sizeof(int), 1, f);
		for (int r = 0; r < row; r++) {
			for (int c = 0; c < col; c++) {
				float x;
				fread(&x, sizeof(float), 1, f);
				set(m->params[i], r, c, x);
			}
		}
	}
	fread(&m->adam_step, sizeof(int), 1, f);
	for (int i = 0; i < param_cnt; i++) {
		int size = storage_size(m->params[i]);
		fread(m->adam_m[i], sizeof(float), size, f);
		fread(m->adam_v[i], sizeof(float), size, f);
	}
	fclose(f);
	return m;
}



//层归一化算子
void layernorm_forward(tensor* out, tensor* x, tensor** gamma, tensor** beta, int y) {
	int n = x->row;
	int d = x->col;
	for (int i = 0; i < n; i++) {
		float mean = 0;
		for (int j = 0; j < d; j++) {
			mean += at(x, i, j);
		}
		mean /= d;
		float var = 0;
		for (int j = 0; j < d; j++) {
			float v = at(x, i, j) - mean;
			var += v * v;
		}
		var /= d;
		float inv = 1.0f / sqrtf(var + sipu);
		for (int j = 0; j < d; j++) {
			float norm = (at(x, i, j) - mean) * inv;
			set(out, i, j, norm * gamma[y]->value[j] + beta[y]->value[j]);
		}
	}

	out->requires_grad = x->requires_grad || gamma[y]->requires_grad || beta[y]->requires_grad;
	if (!out->requires_grad) return;

	out->parent_cnt = 3;
	out->parents = (tensor**)malloc(3 * sizeof(tensor*));
	out->parents[0] = x;
	out->parents[1] = gamma[y];
	out->parents[2] = beta[y];
	out->backward_fn = layernorm_backward;
}

//encoder单头注意力
tensor* encoder_head_attention_forward(tensor* x, encoder_layer* layer, int h) {
	int n = x->row;
	tensor* q = make_tensor(n, dk, dk, 1);
	tensor* k = make_tensor(n, dk, dk, 1);
	tensor* v = make_tensor(n, dk, dk, 1);
	mul(q, x, layer->Wq[h]);
	mul(k, x, layer->Wk[h]);
	mul(v, x, layer->Wv[h]);
	tensor* kt = transpose(k);
	tensor* score = make_tensor(n, n, n, 1);
	mul(score, q, kt);
	tensor* scaled_score = make_tensor(n, n, n, 1);
	scale_tensor(scaled_score, score, 1.0f / sqrtf((float)dk));
	tensor* prob = make_tensor(n, n, n, 1);
	soft_max(prob, scaled_score);
	tensor* out = make_tensor(n, dk, dk, 1);
	mul(out, prob, v);
	return out;
}

//encoder多头注意力
tensor* encoder_multi_head_attention_forward(tensor* x, encoder_layer* layer) {
	int n = x->row;
	tensor* cat = make_tensor(n, dm, dm, 1);
	tensor* head_out[H];
	for (int h = 0; h < H; h++) {
		head_out[h] = encoder_head_attention_forward(x, layer, h);
	}
	cat_heads(cat, head_out);
	tensor* out = make_tensor(n, dm, dm, 1);
	mul(out, cat, layer->Wo);
	return out;
}

//encoder前馈网络
tensor* ffn_forward(tensor* x, encoder_layer* layer) {
	int n = x->row;
	tensor* h1 = make_tensor(n, dff, dff, 1);
	mul(h1, x, layer->W1);
	tensor* a1 = make_tensor(n, dff, dff, 1);
	relu_tensor(a1, h1);
	tensor* out = make_tensor(n, dm, dm, 1);
	mul(out, a1, layer->W2);
	return out;
}

//encoder单层编码器
tensor* encoder_layer_forward(tensor* x, encoder_layer* layer) {
	tensor* n1 = make_tensor(x->row, x->col, x->col, 1);
	layernorm_forward(n1, x, layer->gamma, layer->beta, 0);
	tensor* attn = encoder_multi_head_attention_forward(n1, layer);
	tensor* r1 = make_tensor(x->row, x->col, x->col, 1);
	add(r1, x, attn);
	tensor* n2 = make_tensor(r1->row, r1->col, r1->col, 1);
	layernorm_forward(n2, r1, layer->gamma, layer->beta, 1);
	tensor* ffn = ffn_forward(n2, layer);
	tensor* out = make_tensor(r1->row, r1->col, r1->col, 1);
	add(out, r1, ffn);
	return out;
}

//encoder编码器
tensor* encoder_forward(tensor* x, layer* L) {
	tensor* cur = x;
	for (int i = 0; i < N; i++) {
		tensor* next = encoder_layer_forward(cur, L->layerlist[i]);
		cur = next;
	}
	return cur;
}

//词表查询+注入位置编码
void embedding_with_position(tensor* out, int* list, int n, tensor* E) {
	float t[dm];
	for (int j = 0; j < dm; j += 2) {
		t[j] = powf(10000.0f, (j / 2 * 2) / (float)dm);
		t[j + 1] = t[j];
	}
	for (int i = 0; i < n; i++) {
		int id = list[i];
		for (int j = 0; j < dm; j++) {
			if (j % 2 == 0) {
				set(out, i, j, at(E, id, j) + sinf(i / t[j]));
				continue;
			}
			set(out, i, j, at(E, id, j) + cosf(i / t[j]));
		}
	}

	out->requires_grad = E->requires_grad;
	if (!out->requires_grad) return;

	out->parent_cnt = 1;
	out->parents = (tensor**)malloc(sizeof(tensor*));
	out->parents[0] = E;
	embedding_ctx* ctx = (embedding_ctx*)malloc(sizeof(embedding_ctx));
	ctx->n = n;
	ctx->list = (int*)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) {
		ctx->list[i] = list[i];
	}
	out->ctx = ctx;
	out->free_ctx = free_embedding_ctx;
	out->backward_fn = embedding_backward;
}

//decoder前馈网络
tensor* de_ffn_forward(tensor* x, decoder_layer* layer) {
	int n = x->row;
	tensor* h1 = make_tensor(n, dff, dff, 1);
	mul(h1, x, layer->W1);
	tensor* a1 = make_tensor(n, dff, dff, 1);
	relu_tensor(a1, h1);
	tensor* out = make_tensor(n, dm, dm, 1);
	mul(out, a1, layer->W2);
	return out;
}

//decoder单头交叉注意力
tensor* decoder_crosshead_attention_forward(tensor* x, tensor* encoder_x, decoder_layer* layer, int h) {
	int n = x->row;
	int ne = encoder_x->row;
	tensor* q = make_tensor(n, dk, dk, 1);
	tensor* k = make_tensor(ne, dk, dk, 1);
	tensor* v = make_tensor(ne, dk, dk, 1);
	mul(q, x, layer->Wqc[h]);
	mul(k, encoder_x, layer->Wkc[h]);
	mul(v, encoder_x, layer->Wvc[h]);
	tensor* kt = transpose(k);
	tensor* score = make_tensor(n, ne, ne, 1);
	mul(score, q, kt);
	tensor* scaled_score = make_tensor(n, ne, ne, 1);
	scale_tensor(scaled_score, score, 1.0f / sqrtf((float)dk));
	tensor* prob = make_tensor(n, ne, ne, 1);
	soft_max(prob, scaled_score);
	tensor* out = make_tensor(n, dk, dk, 1);
	mul(out, prob, v);
	return out;
}

//decoder多头交叉注意力
tensor* decoder_multi_crosshead_attention_forward(tensor* x, tensor* encoder_x, decoder_layer* layer) {
	int n = x->row;
	tensor* cat = make_tensor(n, dm, dm, 1);
	tensor* head_out[H];
	for (int h = 0; h < H; h++) {
		head_out[h] = decoder_crosshead_attention_forward(x, encoder_x, layer, h);
	}
	cat_heads(cat, head_out);
	tensor* out = make_tensor(n, dm, dm, 1);
	mul(out, cat, layer->Wo_cross);
	return out;
}

//decoder单头自注意力
tensor* decoder_selfhead_attention_forward(tensor* x, decoder_layer* layer, int h) {
	int n = x->row;
	tensor* q = make_tensor(n, dk, dk, 1);
	tensor* k = make_tensor(n, dk, dk, 1);
	tensor* v = make_tensor(n, dk, dk, 1);
	mul(q, x, layer->Wq[h]);
	mul(k, x, layer->Wk[h]);
	mul(v, x, layer->Wv[h]);
	tensor* kt = transpose(k);
	tensor* score = make_tensor(n, n, n, 1);
	mul(score, q, kt);
	tensor* scaled_score = make_tensor(n, n, n, 1);
	scale_tensor(scaled_score, score, 1.0f / sqrtf((float)dk));
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			set(scaled_score, i, j, -1e9f);
		}
	}
	tensor* prob = make_tensor(n, n, n, 1);
	soft_max(prob, scaled_score);
	tensor* out = make_tensor(n, dk, dk, 1);
	mul(out, prob, v);
	return out;
}

//decoder多头自注意力
tensor* decoder_multi_selfhead_attention_forward(tensor* x, decoder_layer* layer) {
	int n = x->row;
	tensor* cat = make_tensor(n, dm, dm, 1);
	tensor* head_out[H];
	for (int h = 0; h < H; h++) {
		head_out[h] = decoder_selfhead_attention_forward(x, layer, h);
	}
	cat_heads(cat, head_out);
	tensor* out = make_tensor(n, dm, dm, 1);
	mul(out, cat, layer->Wo);
	return out;
}

//decoder单层解码器
tensor* decoder_layer_forward(tensor* x, tensor* encoder_x, decoder_layer* layer) {
	tensor* n1 = make_tensor(x->row, x->col, x->col, 1);
	layernorm_forward(n1, x, layer->gamma, layer->beta, 0);
	tensor* attn = decoder_multi_selfhead_attention_forward(n1, layer);
	tensor* r1 = make_tensor(x->row, x->col, x->col, 1);
	add(r1, x, attn);
	tensor* n2 = make_tensor(r1->row, r1->col, r1->col, 1);
	layernorm_forward(n2, r1, layer->gamma, layer->beta, 1);
	tensor* attn1 = decoder_multi_crosshead_attention_forward(n2, encoder_x, layer);
	tensor* r2 = make_tensor(x->row, x->col, x->col, 1);
	add(r2, r1, attn1);
	tensor* n3 = make_tensor(r2->row, r2->col, r2->col, 1);
	layernorm_forward(n3, r2, layer->gamma, layer->beta, 2);
	tensor* ffn = de_ffn_forward(n3, layer);
	tensor* r3 = make_tensor(x->row, x->col, x->col, 1);
	add(r3, r2, ffn);
	return r3;
}

//decoder解码器
tensor* decoder_forward(tensor* x, tensor* encoder_x, layer* L) {
	tensor* cur = x;
	for (int i = 0; i < N; i++) {
		tensor* next = decoder_layer_forward(cur, encoder_x, L->de_layerlist[i]);
		cur = next;
	}
	return cur;
}

//output解码器
void output_projection(tensor* logits, tensor* x, tensor* W_project) {
	mul(logits, x, W_project);
}

//交叉熵损失
tensor* cross_entropy_loss(tensor* logits, int* targets) {
	tensor* loss = make_tensor(1,1,1,1);
	float Loss = 0;
	tensor* ex = make_tensor(logits->row, logits->col, logits->stride0, logits->stride1);
	for (int i = 0; i < logits->row; i++) {
		float max = at(logits, i, 0);
		for (int j = 1; j < logits->col; j++) {
			if (max < at(logits, i, j))
				max = at(logits, i, j);
		}
		float sum = 0;
		for (int j = 0; j < logits->col; j++) {
			float tmp = expf(at(logits, i, j) - max);
			sum += tmp;
			set(ex, i, j, tmp);
		}
		float tmp = logf(sum) - at(logits, i, targets[i]) + max;
		Loss += tmp;
	}
	Loss /= logits->row;
	loss->value[0] = Loss;

	loss->requires_grad = logits->requires_grad;
	if (!loss->requires_grad) return loss;
	loss->parent_cnt = 1;
	loss->parents = (tensor**)malloc(sizeof(tensor*));
	loss->parents[0] = logits;
	loss->backward_fn = cross_entropy_loss_backward;
	cross_entropy_loss_ctx* ctx = (cross_entropy_loss_ctx*)malloc(sizeof(cross_entropy_loss_ctx));
	ctx->targets = targets;
	ctx->ex = ex;
	loss->ctx = ctx;
	loss->free_ctx = free_cross_entropy_loss_ctx;
	ensure_grad(loss);
	grad_at(loss, 0, 0) = 1;

	return loss;
}


tensor* forward_logits(int* src_list, int len_src_list, int* tgt_list, int len_tgt_list, model* m) {
	tensor* src = make_tensor(len_src_list, dm, dm, 1);
	embedding_with_position(src, src_list, len_src_list, m->src_E);
	tensor* memory = encoder_forward(src, m->layers);
	tensor* tgt = make_tensor(len_tgt_list, dm, dm, 1);
	embedding_with_position(tgt, tgt_list, len_tgt_list, m->tgt_E);
	tensor* dec = decoder_forward(tgt, memory, m->layers);
	tensor* logits = make_tensor(dec->row, m->W_project->col, m->W_project->col, 1);
	output_projection(logits, dec, m->W_project);
	return logits;
}

//训练向前函数
tensor* train_forward(int* src_list, int len_src_list, int* tgt_list0, int* tgt_list1, int len_tgt_list, model* m) {
	tensor* logits = forward_logits(src_list, len_src_list, tgt_list0, len_tgt_list, m);
	tensor* loss = cross_entropy_loss(logits, tgt_list1);
	return loss;
}


//拓扑图收集
void  collect(tensor* node, tensor** order, int* cnt) {
	if (!node->requires_grad) return;
	if (node->mark) return;
	if (node->parent_cnt) {
		for (int i = 0; i < node->parent_cnt; i++) {
			collect(node->parents[i], order, cnt);
		}
	}
	node->mark = 1;
	order[*cnt] = node;
	*cnt += 1;
}

//反向传播循环
void backward(tensor** order, int cnt) {
	for (int i = cnt; i != -1; i--) {
		if (order[i]->backward_fn)
			order[i]->backward_fn(order[i]);
	}
}

int is_param(model* m, tensor* t) {
	for (int i = 0; i < m->param_cnt; i++) {
		if (m->params[i] == t) return 1;
	}
	return 0;
}

void free_graph(tensor** order, int cnt, model* m) {
	for (int i = cnt - 1; i >= 0; i--) {
		if (is_param(m, order[i])) {
			order[i]->mark = 0;
		}
		else {
			free_tensor(order[i]);
		}
	}
	for (int i = 0; i < m->param_cnt; i++) {
		m->params[i]->mark = 0;
	}
}

void free_forward_graph(tensor* root, model* m) {
	tensor* order[2000];
	int cnt = 0;
	collect(root, order, &cnt);
	free_graph(order, cnt, m);
}

//反向传播+释放张量
void train_backward(tensor* root, model* m) {
	tensor* order[2000];
	int cnt = 0;
	collect(root, order, &cnt);
	backward(order, cnt - 1);
	free_graph(order, cnt, m);
}

//梯度更新算子
void sgd_update(tensor* t, float lr) {
	if (!t->requires_grad || !t->grad) return;
	for (int i = 0; i < t->row; i++) {
		for (int j = 0; j < t->col; j++) {
			set(t, i, j, at(t, i, j) - grad_at(t, i, j) * lr);
		}
	}
}

//清空梯度与mark
void zero_grad(tensor* t){
	if (t->grad) {
		int cnt = storage_size(t);
		for (int i = 0; i < cnt; i++) {
			t->grad[i] = 0;
		}
	}
	t->mark = 0;
}

//梯度更新
void update_model(model* m, float lr) {
	for (int i = 0; i < m->param_cnt; i++) {
		sgd_update(m->params[i], lr);
	}
}

void scale_model_grad(model* m, float scale) {
	for (int i = 0; i < m->param_cnt; i++) {
		tensor* t = m->params[i];
		if (!t->grad) continue;
		int size = storage_size(t);
		for (int j = 0; j < size; j++) {
			t->grad[j] *= scale;
		}
	}
}

void adam_update(model* m, float lr) {
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float eps = 0.00000001f;
	m->adam_step += 1;
	float fix1 = 1.0f - powf(beta1, (float)m->adam_step);
	float fix2 = 1.0f - powf(beta2, (float)m->adam_step);
	for (int i = 0; i < m->param_cnt; i++) {
		tensor* t = m->params[i];
		if (!t->grad) continue;
		int size = storage_size(t);
		for (int j = 0; j < size; j++) {
			float g = t->grad[j];
			m->adam_m[i][j] = beta1 * m->adam_m[i][j] + (1.0f - beta1) * g;
			m->adam_v[i][j] = beta2 * m->adam_v[i][j] + (1.0f - beta2) * g * g;
			float now_m = m->adam_m[i][j] / fix1;
			float now_v = m->adam_v[i][j] / fix2;
			t->value[j] -= lr * now_m / (sqrtf(now_v) + eps);
		}
	}
}

void zero_model_grad(model* m) {
	for (int i = 0; i < m->param_cnt; i++) {
		zero_grad(m->params[i]);
	}
}

//单轮训练
float train_step(model* m, int* src_list, int len_src_list, int* tgt_list0, int* tgt_list1, int len_tgt_list, float lr) {
	zero_model_grad(m);
	tensor* loss = train_forward(src_list, len_src_list, tgt_list0, tgt_list1, len_tgt_list, m);
	float loss_value = loss->value[0];
	train_backward(loss, m);
	adam_update(m, lr);
	return loss_value;
}

float train_backward_only(model* m, int* src_list, int len_src_list, int* tgt_list0, int* tgt_list1, int len_tgt_list) {
	tensor* loss = train_forward(src_list, len_src_list, tgt_list0, tgt_list1, len_tgt_list, m);
	float loss_value = loss->value[0];
	train_backward(loss, m);
	return loss_value;
}

struct train_sample {
	int src_len;
	int tgt_len;
	int* src;
	int* tgt_input;
	int* tgt_output;
};

struct train_data {
	int src_vocab;
	int tgt_vocab;
	int sample_cnt;
	train_sample* samples;
};

int is_test_sample(int index) {
	return index % 10 == 0;
}

train_data* load_train_data(const char* path) {
	FILE* f = NULL;
	if (fopen_s(&f, path, "rb") != 0) return NULL;
	train_data* data = (train_data*)malloc(sizeof(train_data));
	fread(&data->src_vocab, sizeof(int), 1, f);
	fread(&data->tgt_vocab, sizeof(int), 1, f);
	fread(&data->sample_cnt, sizeof(int), 1, f);
	data->samples = (train_sample*)malloc(data->sample_cnt * sizeof(train_sample));
	for (int i = 0; i < data->sample_cnt; i++) {
		train_sample* sample = &data->samples[i];
		fread(&sample->src_len, sizeof(int), 1, f);
		fread(&sample->tgt_len, sizeof(int), 1, f);
		sample->src = (int*)malloc(sample->src_len * sizeof(int));
		sample->tgt_input = (int*)malloc(sample->tgt_len * sizeof(int));
		sample->tgt_output = (int*)malloc(sample->tgt_len * sizeof(int));
		fread(sample->src, sizeof(int), sample->src_len, f);
		fread(sample->tgt_input, sizeof(int), sample->tgt_len, f);
		fread(sample->tgt_output, sizeof(int), sample->tgt_len, f);
	}
	fclose(f);
	return data;
}

void free_train_data(train_data* data) {
	for (int i = 0; i < data->sample_cnt; i++) {
		free(data->samples[i].src);
		free(data->samples[i].tgt_input);
		free(data->samples[i].tgt_output);
	}
	free(data->samples);
	free(data);
}

int argmax_row(tensor* t, int row) {
	int best = 0;
	float best_value = at(t, row, 0);
	for (int j = 1; j < t->col; j++) {
		float x = at(t, row, j);
		if (x > best_value) {
			best = j;
			best_value = x;
		}
	}
	return best;
}

int greedy_decode(model* m, train_sample* sample, int* output, int max_len) {
	int tgt_input[32];
	tgt_input[0] = 1;
	int output_len = 0;
	for (int i = 0; i < max_len; i++) {
		tensor* logits = forward_logits(sample->src, sample->src_len, tgt_input, i + 1, m);
		int id = argmax_row(logits, i);
		free_forward_graph(logits, m);
		output[output_len] = id;
		output_len += 1;
		if (id == 2) break;
		tgt_input[output_len] = id;
	}
	return output_len;
}

void print_ids(const char* name, int* ids, int len) {
	printf("%s", name);
	for (int i = 0; i < len; i++) {
		printf(" %d", ids[i]);
	}
	printf("\n");
}

void evaluate_model(model* m, train_data* data, int test_only) {
	float loss_sum = 0;
	int token_cnt = 0;
	int token_correct = 0;
	int teacher_exact = 0;
	int greedy_exact = 0;
	int sample_cnt = 0;
	int error_cnt = 0;

	for (int i = 0; i < data->sample_cnt; i++) {
		if (is_test_sample(i) != test_only) continue;
		sample_cnt += 1;
		train_sample* sample = &data->samples[i];
		tensor* logits = forward_logits(sample->src, sample->src_len, sample->tgt_input, sample->tgt_len, m);
		tensor* loss = cross_entropy_loss(logits, sample->tgt_output);
		loss_sum += loss->value[0];
		int all_correct = 1;
		for (int j = 0; j < sample->tgt_len; j++) {
			int id = argmax_row(logits, j);
			token_cnt += 1;
			if (id == sample->tgt_output[j]) {
				token_correct += 1;
			}
			else {
				all_correct = 0;
			}
		}
		if (all_correct) teacher_exact += 1;
		free_forward_graph(loss, m);

		int output[32];
		int output_len = greedy_decode(m, sample, output, 24);
		int same = output_len == sample->tgt_len;
		for (int j = 0; j < output_len && same; j++) {
			if (output[j] != sample->tgt_output[j]) same = 0;
		}
		if (same) greedy_exact += 1;
		if (test_only && !same && error_cnt < 3) {
			printf("held-out error sample %d\n", i);
			print_ids("target:", sample->tgt_output, sample->tgt_len);
			print_ids("greedy:", output, output_len);
		}
		if (test_only && !same) error_cnt += 1;
		if (sample_cnt <= 3) {
			printf("sample %d\n", i);
			print_ids("target:", sample->tgt_output, sample->tgt_len);
			print_ids("greedy:", output, output_len);
		}
	}

	float average_loss = loss_sum / sample_cnt;
	printf("\nmodel metrics\n");
	printf("samples = %d\n", sample_cnt);
	printf("average cross entropy = %f\n", average_loss);
	printf("perplexity = %f\n", expf(average_loss));
	printf("teacher token accuracy = %f\n", (float)token_correct / token_cnt);
	printf("teacher exact sentence accuracy = %f\n", (float)teacher_exact / sample_cnt);
	printf("greedy exact sentence accuracy = %f\n", (float)greedy_exact / sample_cnt);
	printf("\nuniform baseline\n");
	printf("cross entropy = %f\n", logf((float)data->tgt_vocab));
	printf("perplexity = %d\n", data->tgt_vocab);
	printf("token accuracy = %f\n", 1.0f / data->tgt_vocab);
}

int main(int argc, char** argv) {
	#ifdef _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	#endif
	srand(1);
	const char* data_path = argc > 1 ? argv[1] : "data/mini_en_zh.bin";
	const char* model_path = argc > 2 ? argv[2] : "data/mini_model.bin";
	int train_steps = argc > 3 ? atoi(argv[3]) : 10000;
	int resume = argc > 4 ? atoi(argv[4]) : 0;
	train_data* data = load_train_data(data_path);
	if (!data) {
		printf("cannot open %s\n", data_path);
		return 1;
	}

	int* train_indices = (int*)malloc(data->sample_cnt * sizeof(int));
	int train_cnt = 0;
	for (int i = 0; i < data->sample_cnt; i++) {
		if (!is_test_sample(i)) {
			train_indices[train_cnt] = i;
			train_cnt += 1;
		}
	}
	printf("train samples = %d, test samples = %d\n", train_cnt, data->sample_cnt - train_cnt);

	model* m = resume ? load_model(model_path) : make_model(data->src_vocab, data->tgt_vocab);
	if (!m) {
		printf("cannot open %s\n", model_path);
		free(train_indices);
		free_train_data(data);
		return 1;
	}
	printf("parameter tensors = %d\n", m->param_cnt);
	clock_t train_begin = clock();
	const int batch_size = 8;
	float loss_sum = 0;
	for (int i = 0; i < train_steps; i++) {
		zero_model_grad(m);
		for (int b = 0; b < batch_size; b++) {
			train_sample* sample = &data->samples[train_indices[next_train_random() % train_cnt]];
			loss_sum += train_backward_only(m, sample->src, sample->src_len, sample->tgt_input, sample->tgt_output, sample->tgt_len);
		}
		scale_model_grad(m, 1.0f / batch_size);
		float lr = 0.0003f;
		if (m->adam_step < 2000) {
			lr *= (float)(m->adam_step + 1) / 2000.0f;
		}
		adam_update(m, lr);
		if ((i + 1) % 500 == 0) {
			printf("update %d, average loss = %f\n", i + 1, loss_sum / (500 * batch_size));
			loss_sum = 0;
			fflush(stdout);
		}
		if ((i + 1) % 1000 == 0) {
			save_model(m, model_path);
		}
	}
	float train_seconds = (float)(clock() - train_begin) / CLOCKS_PER_SEC;
	printf("training time = %f seconds\n", train_seconds);
	if (train_steps > 0) {
		printf("time per update = %f ms\n", train_seconds * 1000.0f / train_steps);
		printf("time per sample = %f ms\n", train_seconds * 1000.0f / (train_steps * batch_size));
	}
	save_model(m, model_path);
	printf("saved %s\n", model_path);
	free_model(m);
	m = load_model(model_path);
	if (!m) {
		printf("cannot open %s\n", model_path);
		free(train_indices);
		free_train_data(data);
		return 1;
	}
	printf("\nevaluate loaded model on train split\n");
	clock_t evaluate_begin = clock();
	evaluate_model(m, data, 0);
	printf("\nevaluate loaded model on held-out split\n");
	evaluate_model(m, data, 1);
	float evaluate_seconds = (float)(clock() - evaluate_begin) / CLOCKS_PER_SEC;
	printf("evaluation time = %f seconds\n", evaluate_seconds);
	free_model(m);
	free(train_indices);
	free_train_data(data);
	#ifdef _DEBUG
	if (_CrtDumpMemoryLeaks()) {
		printf("memory leak detected\n");
	}
	else {
		printf("no memory leak detected\n");
	}
	#endif
	return 0;
}
