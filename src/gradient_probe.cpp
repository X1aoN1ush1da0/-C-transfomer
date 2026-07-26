#define main transformer_program_main
#include "main.cpp"
#undef main

void print_grad_stats(const char* name, tensor* t) {
	if (!t->grad) {
		printf("%s: no gradient\n", name);
		return;
	}
	int size = storage_size(t);
	double square_sum = 0;
	double abs_sum = 0;
	float max_abs = 0;
	for (int i = 0; i < size; i++) {
		float x = t->grad[i];
		float ax = fabsf(x);
		square_sum += x * x;
		abs_sum += ax;
		if (ax > max_abs) max_abs = ax;
	}
	printf("%s: l2=%e mean_abs=%e max_abs=%e\n", name,
		sqrt(square_sum), abs_sum / size, max_abs);
}

void print_value_stats(const char* name, tensor* t) {
	int size = storage_size(t);
	double abs_sum = 0;
	float min_value = t->value[0];
	float max_value = t->value[0];
	for (int i = 0; i < size; i++) {
		float x = t->value[i];
		abs_sum += fabsf(x);
		if (x < min_value) min_value = x;
		if (x > max_value) max_value = x;
	}
	printf("%s: mean_abs=%e min=%e max=%e\n", name, abs_sum / size, min_value, max_value);
}

int main() {
	train_data* data = load_train_data("data/tatoeba_medium_simplified.bin");
	model* m = load_model("data/tatoeba_medium_adam.bin");
	train_sample* sample = &data->samples[1];
	zero_model_grad(m);
	tensor* loss = train_forward(sample->src, sample->src_len,
		sample->tgt_input, sample->tgt_output, sample->tgt_len, m);
	printf("loss=%f\n", loss->value[0]);
	train_backward(loss, m);

	print_grad_stats("src embedding", m->src_E);
	print_grad_stats("encoder layer 0 Wq", m->layers->layerlist[0]->Wq[0]);
	print_grad_stats("encoder layer 0 W1", m->layers->layerlist[0]->W1);
	print_grad_stats("encoder layer 5 Wq", m->layers->layerlist[5]->Wq[0]);
	print_grad_stats("encoder layer 5 W1", m->layers->layerlist[5]->W1);
	print_value_stats("encoder layer 0 gamma 0", m->layers->layerlist[0]->gamma[0]);
	print_value_stats("encoder layer 5 gamma 1", m->layers->layerlist[5]->gamma[1]);
	print_grad_stats("tgt embedding", m->tgt_E);
	print_grad_stats("decoder layer 0 Wq", m->layers->de_layerlist[0]->Wq[0]);
	print_grad_stats("decoder layer 0 Wqc", m->layers->de_layerlist[0]->Wqc[0]);
	print_grad_stats("decoder layer 0 W1", m->layers->de_layerlist[0]->W1);
	print_grad_stats("decoder layer 5 Wq", m->layers->de_layerlist[5]->Wq[0]);
	print_grad_stats("decoder layer 5 Wqc", m->layers->de_layerlist[5]->Wqc[0]);
	print_grad_stats("decoder layer 5 W1", m->layers->de_layerlist[5]->W1);
	print_value_stats("decoder layer 0 gamma 0", m->layers->de_layerlist[0]->gamma[0]);
	print_value_stats("decoder layer 5 gamma 2", m->layers->de_layerlist[5]->gamma[2]);
	print_grad_stats("output projection", m->W_project);

	free_model(m);
	free_train_data(data);
	return 0;
}
