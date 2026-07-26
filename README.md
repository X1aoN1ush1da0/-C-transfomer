# 用C语言从零手写transformer

这是我一个的练手项目，纯手搓。主要目的是为了学习transformer的底层知识并且练习C语言。

本项目用C语言完成了：

- 所有公式的数学推导
- 所有transformer所需的向前传播算子
- 完整的transformer向前传播的流程
- 所有transformer所需的反向传播算子
- 一套足够用的自动微分系统
- 一套稳定的内存管理系统

总的来讲，这是一个高度定制化但是很完整的transformer项目，读者可以在其中学到不少的机器学习和数学知识。
(需要前备知识包括：基本的高数和线性代数：会求偏导，会算矩阵的乘法，能够理解线性变化。能看懂基本的C语言语法)

## transformer介绍

“Transformer 是一种用于自然语言处理（NLP）和其他“序列到序列任务”的深度学习模型架构。它由 Vaswani 等人在 2017 年提出，并通过引入自注意力（Self-Attention Mechanism） 和 并行计算，彻底改变了 NLP 领域的研究和应用。”

通俗来讲，transformer主要由编码器和解码器构成，而这两个者差别不大，它们的核心都是“注意力”，可以理解为，编码器接受了外部的输入语句，然后通过“自注意力”来计算出一个浓缩了输入语句的意义的一个“编码”，然后解码器再同时参考这个“编码”和它已生成的语句，同时做“交叉注意力”和“自注意力”，最后来预测出下一个词。用翻译任务来类比，就是说编码器先总结出原句的意思（以一种人类不可名状的高浓度数据块总结），然后解码器同时参考这个“高浓度数据块”和自己“刚刚才翻译的那些语句”，来预测出下一个词究竟该是什么。
<div  align="center">
<img src=assets/process.png width=100% alt=process>
</div>

## tensor张量

计算机中的“张量”就是一个矩阵。它是机器学习中用来存放数据的“容器”。由于计算机并不能直接在内存中画个方格来当成一个矩阵，所以我们只能用连续的内存来当作一个张量，但是这样就有一个问题，我们得到的就相当于是一个一维的连续列表，而不是一个二维的矩阵。为了解决这个问题，引入一个参数—— $stride$ （步长）:

```
struct tensor {
	float* value;
	int row;  //行数
	int col;  //列数
	int stride0;  //行步长
	int stride1;  //列步长
}
```

这里定义了一个 $tensor$ 机构体。其中的 $value$ 和 $row$ 、 $col$ 很好理解， $value$ 就是一个指向数组起点的一个指针，而 $row$ 和 $col$ 则记录的矩阵的行和列。<br>
而对于 $stride0$ 和 $stride1$ ，
以 $stride0$ 为例，它的意思就是当你想从一个矩阵中的坐标（譬如（2，3））跳到“下一行”（（3，3））时，在一维的连续内存中需要跳过多少个数，其核心目的就是我们在定位张量中的某个值时，就可以直接用 $stride$ 来辅助定位:

```
float at(tensor* a, int i, int j) {
	return a->value[i * a->stride0 + j * a->stride1];
}
//a->b 等价于 (*a).b , 而整个代码中我们都是用的tensor* a，也就是用指针来追踪张量
```

这个函数就可以通过输入张量a、目标位置的坐标（i,j），就可以返回查询到的值。
步长的设计还有一个好处，就是在做矩阵转置的时候会很轻松：

```
tensor* transpose(tensor* a) {
	tensor* out = make_tensor(a->value, a->col, a->row, a->stride1, a->stride0); //创建张量的函数
    return out;
}
```

可以看到，我们只用把原来的列表初始位的指针给拿过来，然后行和列交换，并且将步长给交换，就可以完美的适用于前面的查询函数了（具体原理可以自行验证）。这样相当于没有做拷贝等大额开销。
下面展示的是建立张量的函数：

```
tensor* make_tensor(int row, int col, int stride0, int stride1) {
	tensor* t = (tensor*)malloc(sizeof(tensor));
	t->row = row;
	t->col = col;
	t->stride0 = stride0;
	t->stride1 = stride1;
	int size = (row - 1) * stride0 + (col - 1) * stride1 + 1;
	t->value = (float*)malloc(size * sizeof(float));  //分配内存空间
    return t;
}
```

该函数指定了张量的形状和步长，并且给其value分配了内存空间。

## 向前传播算子

在向前传播的过程中需要做各种矩阵运算，所以需要将各种算子封装好，如：

```
加法算子：
void add(tensor* c, tensor* a, tensor* b) {
	int size = a->row * a->col;
	for (int i = 0; i < size; i++) {
		c->value[i] = a->value[i] + b->value[i];
	}
}

乘法算子：
void mul(tensor* c, tensor* a, tensor* b) {
	int p = a->row;
	int q = a->col;
	int r = b->col;
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
```

以乘法算子为例，相乘的两个张量是a和b，而用来承接结果的是张量c。随后写循环，以此算出c(i,j)的值。
当然也有一些比较复杂的算子，如softmax和层归一化算子等:

```
softmax算子：
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
}
```

这里的softmax是机器学习中十分常见的“输出层激活函数”，它的计算规则是：

$$
y_i=\frac{e^{x_i}}{\sum e^x}
$$

也就是将整个矩阵的值都在下面套一个$e^x$，然后每一行分别累加起来并且除这个累加值。这样的作用就是可以将整行数都归一化，可以用来输出概率分布

## 注意力头

“注意力”是transformer最核心的部分。下面是自注意力的原理：
每一个注意力头有：

- 一个输入矩阵：$\mathbf X$ ，形状是 $n\times d_{modle}$ ，其中 $n$ 是“输入语句”的长度，之所以有$n$行是因为整个输入矩阵是由多个词的词向量按行拼成的。
- 三个权重矩阵：$\mathbf W_q$ 、 $\mathbf W_k$ 、$\mathbf W_v$ ,它们的形状都是 $d_{modle} \times d_k$ ，其中$d_{modle}$和$d_k$都是和模型大小有关的固定值。
<div  align="center">
<img src=assets/qkv.png width=50% height=60% alt=qkv>
</div>
  具体的计算过程是这样的：

```
tensor* encoder_head_attention_forward(tensor* x, encoder_layer* layer, int h) {
	int n = x->row;
	tensor* q = make_tensor(n, dk, dk, 1);
	tensor* k = make_tensor(n, dk, dk, 1);
	tensor* v = make_tensor(n, dk, dk, 1);
	mul(q, x, layer->Wq[h]);  //乘法算子
	mul(k, x, layer->Wk[h]);
	mul(v, x, layer->Wv[h]);
	tensor* kt = transpose(k);
	tensor* score = make_tensor(n, n, n, 1);
	mul(score, q, kt);
	tensor* scaled_score = make_tensor(n, n, n, 1);
	scale_tensor(scaled_score, score, 1.0f / sqrtf((float)dk));  //扩倍算子
	tensor* prob = make_tensor(n, n, n, 1);
	soft_max(prob, scaled_score);  //softmax算子
	tensor* out = make_tensor(n, dk, dk, 1);
	mul(out, prob, v);
	return out;
}
```

其中，Q的含义是query（查询），K是key（键），V是value（值）。
翻译成自然语言就是：将$\mathbf X$分别和三个权重矩阵相乘，也就是分别计算这某一个词需要什么知道东西（Q）、它能提供什么东西（K）、它能提供的具体内容是什么（V）。

*（整个x矩阵是由一个一个词按行平成的，而实际上每一个词（一行）都可以独立的与qkv矩阵相乘计算，这里直接用整个X矩阵去乘，结果是相同的）*

乘完之后得到结果$Q_h$、$K_h$、$V_h$，然后计算公式：

$$
out=softmax(\frac{Q_h{K_h}^T}{\sqrt{d_{modle}}})V_h
$$

其中:

- $Q_h{K_h}^T$ 的目的是用每一个词的Q向量去“查询”每一个词的K向量，转置的目的是为了能让两矩阵形状可以相乘。而乘得的结果的第i行第j列的含义就是第i个词和第j个词的关联程度。
- 除以$\sqrt{d_{modle}}$的目的是为了防止“点积方差随维度爆炸”（实际上$\sqrt{d_{modle}}$是QK矩阵的标准差），这样就可以避免softmax中出现极大和极小值导致概率两极化（极端偏向1和0，会导致梯度消失）。
- 套softmax的意义就是概率归一化，将QK的结果彻底变成每两个词之间的关联概率
- 最后乘V矩阵，相当于用前面算出的概率来乘每一个词的实际含义表，最后就得到了单头的注意力结果，也就是模型对于这段输入的理解
- out矩阵的形状是 $d_{modle}\times d_k$ .

## 多头注意力

前面我们实现的是单头注意力，仅使用一组 QKV 矩阵，只能学习单一的词语关联模式。
而标准 Transformer 每层采用多头注意力，也就是有多个注意力头，来并行提取不同维度的文本信息。<br>
<div  align="center">
<img src=assets/multi%20head.png width=50% alt=multi_head >
</div>
<br>
譬如，对同一句话，有的头负责主谓、冠词修饰的局部语法关系，有的头负责代词与前文名词的指代联系，还有的头负责匹配跨分句的长距离语义依赖。多个头各司其职，分别建模不同类型的语义关联，最后将所有头的特征拼接融合，极大拓展了模型捕捉复杂语言关系的能力。
而这里的操作很直接，即直接将每个头的结果并排拼接起来，然后使用一个投影矩阵$W_o$，这个投影矩阵唯一的目的就是对拼接起来的矩阵做线性变换，相当于是让每个头的不同维度的语义信息相融合，得到更全面的整句信息：
<div  align="center">
<img src=assets/attn%20head%20project.png width=80% alt=attn_head>
</div><br>


```
多头注意力：
tensor* encoder_multi_head_attention_forward(tensor* x, encoder_layer* layer) {
	int n = x->row;
	tensor* cat = make_tensor(n, dm, dm, 1);
	tensor* head_out[H];  //H是“注意力头”数
	for (int h = 0; h < H; h++) {
		head_out[h] = encoder_head_attention_forward(x, layer, h);  //分别计算每一个头
	}
	cat_heads(cat, head_out);  //张量拼接算子
	tensor* out = make_tensor(n, dm, dm, 1);
	mul(out, cat, layer->Wo);  //乘投影算子来混合信息
	return out;
}
```

*实际上有关系：“ $d_{modle}=H\times d_k$ ”，这使得多头拼接后的矩阵和输入是相同的*

## 编码器
<div  align="center">
<img src=assets/encoder.png width=60% alt=encoder>
</div>
<br>
实际上，一个编码器是由多层构成的，相当于是串联了多个单层编码器，当然，这每层的编码器的所有权重都是不相同的。我们需要做一些管理：

```
//单层管理
struct encoder_layer {
	tensor* Wq[H];
	tensor* Wk[H];
	tensor* Wv[H];  //这三个是前面的每一个头的QKV矩阵的指针列表
	tensor* Wo;  //这是用来处理多头注意力结果的投影矩阵
	tensor* gamma[2];
	tensor* beta[2];//这两个是层归一化的权重，即将讲到
	tensor* W1;
	tensor* W2;//这两个是前馈神经网络的权重，即将讲到
};

//多层管理
struct layer {
	encoder_layer* layerlist[N];  //单层编码器的指针列表
	decoder_layer* de_layerlist[N];  //这是后面的解码器
};
```

### 层归一化：

层归一化是将每一层分别做可学习的归一化，公式为：



其中，$\mu $ 是该行的平均值，而 $\sigma$ 是该行的标准差；$\epsilon$是一个极小的常数，用来防止分母变成0；而$\gamma$和$\beta$是可学习的参数矩阵（$1\times d_{modle}$），即层归一化的每一行都会对应一个不同的系数）。
代码如下：

```
层归一化算子：
void layernorm_forward(tensor* out, tensor* x, tensor** gamma, tensor** beta, int y) {  //由于一层模型有多个层归一化，且它们的权重是不同的，故用y来标记是哪一层
	int n = x->row;
	int d = x->col;
	for (int i = 0; i < n; i++) {
		float mean = 0;
		for (int j = 0; j < d; j++) {
			mean += at(x, i, j);
		}
		mean /= d;  //算出均值
		float var = 0;
		for (int j = 0; j < d; j++) {
			float v = at(x, i, j) - mean;
			var += v * v;
		}
		var /= d;  //算出方差
		float inv = 1.0f / sqrtf(var + sipu);
		for (int j = 0; j < d; j++) {
			float norm = (at(x, i, j) - mean) * inv;
			set(out, i, j, norm * gamma[y]->value[j] + beta[y]->value[j]);
		}  //配上可学习参数
	}
}
```

### ffn前馈神经网络

前馈神经网络指的是“信号单向向前，无环、无反馈”的一个网络，很多结构如mlp、cnn等都属于前馈神经网络。而在transformer里面，前馈神经网络指的就是全连接多层感知器（mlp），mlp是最基本的神经网络，它的每一层神经元都与下一层的每个神经元相连接，按照一定权重传播信号，并且每一层有激活函数，用来引入非线性。数学公式可以表示为：

$$
1、z=\mathbf W x+b\quad //全连接\\
2、某种激活函数，如：y=relu(z)
$$

```
前馈神经网络：
tensor* ffn_forward(tensor* x, encoder_layer* layer) {
	int n = x->row;
	tensor* h1 = make_tensor(n, dff, dff, 1);  //dff是一个定好的常数，通常大于dm,相当于是扩大特征容量
	mul(h1, x, layer->W1);
	tensor* a1 = make_tensor(n, dff, dff, 1);
	relu_tensor(a1, h1);
	tensor* out = make_tensor(n, dm, dm, 1);  //dm用来恢复原有输入的大小
	mul(out, a1, layer->W2);
	return out;
}
```

因为每一个词向量都是横向的，所以此处的fnn其实完全是在分别处理每一个词向量，它的任务相当于是加工、升级单个token的特征，与前面的注意力头专攻多token关系互补。
<div  align="center">
<img src=assets/ffn.png width=60% alt=ffn>
</div><br>


### 残差连接

残差连接是一个机器学习中很有用的“插件”，其公式是：

$$
x_{out}=x+\mathcal F(x)
$$

其中，$\mathcal F(x)$ 其实就是 $x_{out}-x$ 的残差值，这样的好处有：

- 防止梯度消失：始终会有一个梯度“1”向前传递
- 简化学习目标，加速模型收敛：
  让模型学习 $\mathcal F(x)$ ，即中间的残差部分，只需要简单修正原有的特征
- 保留浅层特征：
  直接保留输入$x$，让浅层语义可以顺利传递到深层
<div  align="center">
<img src=assets/res.png width=80% alt=res>
</div>

下面是transformer单层解码器的完整流程：

```
单层编码器：
tensor* encoder_layer_forward(tensor* x, encoder_layer* layer) {
	tensor* attn = encoder_multi_head_attention_forward(x, layer);  //多头注意力
	tensor* r1 = make_tensor(x->row, x->col, x->col, 1);
	add(r1, x, attn);  //残差连接1
	tensor* n1 = make_tensor(r1->row, r1->col, r1->col, 1);
	layernorm_forward(n1, r1, layer->gamma, layer->beta, 0);  //层归一化1
	tensor* ffn = ffn_forward(n1, layer);  //前馈神经网络
	tensor* r2 = make_tensor(n1->row, n1->col, n1->col, 1);
	add(r2, n1, ffn);  //残差连接2
	tensor* out = make_tensor(r2->row, r2->col, r2->col, 1);
	layernorm_forward(out, r2, layer->gamma, layer->beta, 1);  //层归一化2
	return out;
}
```

完整的编码器就是简单的将单层给堆叠起来：

```
encoder编码器：
tensor* encoder_forward(tensor* x, layer* L) {
	tensor* cur = x;  //起始为第一层
	for (int i = 0; i < N; i++) {  //N代表层数
		tensor* next = encoder_layer_forward(cur, L->layerlist[i]);
		cur = next;  //每一层的输出作为下一层的输入
	}
	return cur;
}
```

## 词嵌入

输入模型的语句是离散的文本字符，但是编码器接受的是一个由词行向量组成的矩阵。所以，在模型的开头有一个“词嵌入”模块，它负责将输入的语句(token)给转换成编码器可以接受的词向量（token向量）矩阵。
<div  align="center">
<img src=assets/embed.png width=100% alt=embed>
</div>

### token向量映射

模型将token映射成token向量，而token向量之间是有规律的，譬如：

1. 语义相近的 token，向量空间距离更近：
   猫” 和 “猫咪”、“小狗” 和 “狗子” 的向量欧氏距离更小；而 “猫” 与 “汽车” 编码词义相似度。
2. 向量支持线性语义运算:
   向量存在可解释的代数关系，如：$国王 - 男人 + 女人 ≈ 王后$，向量维度会包含性别、身份、物种、动作等语义特征。

现代大模型的token向量都是随机初始化，然后和整个模型一起更新的，经过长时间训练后每个token会找到自己最合适的位置。而一些小模型可能会使用预训练好的token向量。

### 正弦位置编码（positional encoding）

注意力机制的一大缺陷就是它将所有的token一视同仁，然而这会导致丢失原有的语句中的位置信息，在他眼里“狗咬人”和“人咬狗”就是没有区别的。因此，引入位置编码：

$$
PE_{(pos,2i)}=\mathrm{sin}(\frac{pos}{10000^{\frac{2i}{d_{modle}}}})\\[12pt]
PE_{(pos,2i+1)}=\mathrm{cos}(\frac{pos}{10000^{\frac{2i}{d_{modle}}}})
$$

其具体的使用方法就是直接在每个位置上面加上这个数。

下面是完整的词嵌入代码：

```
词嵌入：
void embedding_with_position(tensor* out, int* list, int n, tensor* E) {
	float t[dm];
	for (int j = 0; j < dm; j += 2) {
		t[j] = powf(10000.0f, (j / 2 * 2) / (float)dm);  //预计算以节省性能
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
}
```

## 解码器

输入模型的语句在经过词嵌入、编码器后，随即就会进入解码器。解码器在形式上和编码器非常相似，唯一的区别就是它有额外的“交叉注意力”和“掩码”。

### 交叉注意力

交叉注意力头几乎和自注意力头一模一样，唯一的区别就是它的的KV权重矩阵乘的是编码器的结果$x_{encoder}$，而Q矩阵则乘的是当前的$x_{input}$。相当于是用Q矩阵来查询现在的语句需要什么，然后用K矩阵去到编码器输出中去找原句中能提供什么，然后在原句中把找到的对应语义给拎出来。下面是交叉注意力的代码：

```
//decoder单头交叉注意力
tensor* decoder_crosshead_attention_forward(tensor* x, tensor* encoder_x, decoder_layer* layer, int h) {
	int n = x->row;
	int ne = encoder_x->row;
	tensor* q = make_tensor(n, dk, dk, 1);
	tensor* k = make_tensor(ne, dk, dk, 1);
	tensor* v = make_tensor(ne, dk, dk, 1);
	mul(q, x, layer->Wqc[h]);  //乘的是本函数的输入
	mul(k, encoder_x, layer->Wkc[h]);  //乘的是编码器的输出
	mul(v, encoder_x, layer->Wvc[h]);  //乘的是编码器的输出
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
```

### mask掩码
（掩码只在训练时启用）<br>

掩码是在自注意力（除了掩码，和前面编码器的注意力头完全一样）中使用的，，它是将 $\mathbf Q \mathbf V^T$ 的整个上三角部分给全部填充一个非常大的负数，这样一来在经过softmax之后，所有的上三角部分就会几乎变成0（$e^x$函数的特性），前面讲过，$softmax(\frac{Q_h{K_h}^T}{\sqrt{d_{modle}}})$的第i行第j列的值的含义就是原句中第i个词和第j个词直接的关联程度，而将上三角部分全部变为0，相当于是让模型不能提前看到答案，也就是说一个词只能去做自己前面词的自注意力，这样就模拟了在实际生成时的推理方式。
<div  align="center">
<img src=assets/mask%20v.png width=90% alt=mask>
</div><br>


完整的decoder自注意力头:
```
decoder自注意力头：
tensor* decoder_selfhead_attention_forward(tensor* x, decoder_layer* layer, int h) {
	int n = x->row;
	tensor* q = make_tensor(n, dk, dk, 1);
	tensor* k = make_tensor(n, dk, dk, 1);
	tensor* v = make_tensor(n, dk, dk, 1);
	mul(q, x, layer->Wq[h]);
	mul(k, x, layer->Wk[h]);
	mul(v, x, layer->Wv[h]);  //正常的自注意力算法
	tensor* kt = transpose(k);
	tensor* score = make_tensor(n, n, n, 1);
	mul(score, q, kt);
	tensor* scaled_score = make_tensor(n, n, n, 1);
	scale_tensor(scaled_score, score, 1.0f / sqrtf((float)dk));
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			set(scaled_score, i, j, -1e9f);
		}  //布置掩码
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
```

### 解码器
解码器的具体构造：

```
//decoder单层解码器
tensor* decoder_layer_forward(tensor* x, tensor* encoder_x, decoder_layer* layer) {
	tensor* attn = decoder_multi_selfhead_attention_forward(x, layer);
    //多头自注意力
	tensor* r1 = make_tensor(x->row, x->col, x->col, 1);
	add(r1, x, attn);
    //残差连接
	tensor* n1 = make_tensor(r1->row, r1->col, r1->col, 1);
	layernorm_forward(n1, r1, layer->gamma, layer->beta, 0);
    //层归一化
	tensor* attn1 = decoder_multi_crosshead_attention_forward(n1, encoder_x, layer);
    //多头交叉注意力
	tensor* r2 = make_tensor(x->row, x->col, x->col, 1);
	add(r2, n1, attn1);
    //残差连接
	tensor* n2 = make_tensor(r2->row, r2->col, r2->col, 1);
	layernorm_forward(n2, r2, layer->gamma, layer->beta, 1);
    //层归一化
	tensor* ffn = de_ffn_forward(n2, layer);
    //前馈神经网络
	tensor* r3 = make_tensor(x->row, x->col, x->col, 1);
	add(r3, n2, ffn);
    //残差连接
	tensor* out = make_tensor(r3->row, r3->col, r3->col, 1);
	layernorm_forward(out, r3, layer->gamma, layer->beta, 2);
    //层归一化
	return out;
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
```

之所以将自注意力放在交叉注意力的前面，是因为我们希望模型能够先摸清楚句子中的各种抽象关系，然后再去看对应的编码。
## 损失函数

decoder生成出的矩阵就进入最后的模块来计算损失：

### output解码器

```
output解码器：
void output_projection(tensor* logits, tensor* x, tensor* W_project) {
	mul(logits, x, W_project);
}
```

x 是指向解码器输出矩阵的指针，其矩阵的形状仍为 $n\times d_{modle}$ ；W_project在经典做法中是直接复用词表矩阵E（需要进行转置），其维度是 $d_{modle}\times d_{list} $（$d_{list}$是原词表的长度，相当于是包含的总token数）。故投影结果第i行第j列的含义可以理解为:模型预测的第i个词是词表中第j个词之间的“概率”。

### 交叉熵损失

交叉熵是机器学习中非常常见的用于分类任务的损失函数，其公式是：

$$
Loss=\sum -y_i\mathrm{log}(\hat{y_i})
$$

其中 $y_i$ 是真实标签（相当于标准答案），$\hat{y_i}$ 是模型预测的标签（相当于是模型给出的答案）。而 $Loss$ 越小，说明模型预测能力越强。
而在transformer中的交叉熵是这样的：

```
交叉熵损失:
tensor* cross_entropy_loss(tensor* logits, int* targets) {
	tensor* loss = make_tensor(1,1,1,1);
	float Loss = 0;
	tensor* ex = make_tensor(logits->row, logits->col, logits->stride0, logits->stride1);
	for (int i = 0; i < logits->row; i++) {
		float max = at(logits, i, 0);
		for (int j = 1; j < logits->col; j++) {
			if (max < at(logits, i, j))
				max = at(logits, i, j);
		}  //找出每行的最大值
		float sum = 0;
		for (int j = 0; j < logits->col; j++) {
			float tmp = expf(at(logits, i, j) - max);  //减去最大值后套e^x
			sum += tmp;
		}
		float tmp = logf(sum) - at(logits, i, targets[i]) + max;
		Loss += tmp;
	}
	Loss /= logits->row;
	loss->value[0] = Loss;
	return loss;
}
```

它的算法其实是对softmax之后的结果进行交叉熵损失：首先是找到每一行的最大值，然后让这一行数都减去这个最大值，随后进行softmax（这样可以减小$e^x$的计算量避免越界,且不会改变原有概率）。然后再直接用标准答案中对应的那个词的位置定位到softmax之后的那个矩阵的对应位置并且提取其概率来进行交叉熵的计算。结合公式可以表示为：

$$
答案的词的对应位置是targets[i]\\[12pt]
直接找到（i,targets[i]）的位置\\[12pt]
其概率为p_{(i,targets[i])}=\frac{e^{x_{(i,tragets[i])}-max}}{\sum e^{x_{(i,j)}-max}}\\[12pt]
由于真实标签为 1\\[12pt]
故Loss=-1\times\mathrm{log}(p_{(i,targets[i])})=\mathrm{log}(\sum e^{x-max})-x_{(i,tragets[i])}+max
$$

## 总向前传播
<div  align="center">
<img src=assets/modle.png width=100% alt=modle>
</div>

```
训练向前函数：
tensor* train_forward(int* src_list, int len_src_list, int* tgt_list0, int* tgt_list1, int len_tgt_list, model* m) {
	tensor* src = make_tensor(len_src_list, dm, dm, 1);
	embedding_with_position(src, src_list, len_src_list, m->src_E);
    //（原句）词嵌入
	tensor* memory = encoder_forward(src, m->layers);
    //编码器
	tensor* tgt = make_tensor(len_tgt_list, dm, dm, 1);
	embedding_with_position(tgt, tgt_list0, len_tgt_list, m->tgt_E);
    //（答案）词嵌入
	tensor* dec = decoder_forward(tgt, memory, m->layers);
    //解码器
	tensor* logits = make_tensor(dec->row, m->W_project->col, m->W_project->col, 1);
	output_projection(logits, dec, m->W_project);
    //output投影
	tensor* loss = cross_entropy_loss(logits, tgt_list1);
    //交叉熵
	return loss;
}
```

值得注意的是：

1. 编码器只用接受输入的“原句”，而解码器接受的是“答案”和编码器输出的“编码”。
2. 函数传入了tgt_list0和tgt_list1，这主要是为了方便，譬如，list0是[\<SOS>, 狗, 爱,猫]，而list1就会是[狗, 爱, 猫, \<EOS>]，这里的\<SOS>是序列起始符，而\<EOS>是序列结束符，前者用来让解码器能够有一个“起始位点”，而后者则是“终止密码子”。

## 张量改造

我们在正向传播中一路向前计算，得到了很多中间量和最后的loss，接下来要做的就是反向传播来传递梯度，从而对权重进行优化，为了配套自动微分，需要先对张量结构进行改造：

```
struct tensor {
	float* value;
	int row;
	int col;
	int stride0;
	int stride1;
	int value_owner;  //用于管理转置归属的参数

	float* grad;  //用于储存梯度，和value的长度相同
	int requires_grad;  //记录是否需要更新梯度
	int parent_cnt;  //记录有几个父节点
	struct tensor** parents;  //记录父节点指针
	void(*backward_fn)(struct tensor* self);  //记录该节点的反向算子函数
	void* ctx;  //用于储存高消耗的中间量，节约性能
	void(*free_ctx)(void* ctx);  //用于记录ctx释放函数
	int mark;  //记录该节点是否需要记录在图中
};
```

同时，我们在创建张量时也需要同步建立这些信息：

```
//在建立新张量时需要指定更多的参数
tensor* make_tensor_view(float* value, int row, int col, int stride0, int stride1) {
	tensor* t = (tensor*)malloc(sizeof(tensor));
	t->value = value;
	t->row = row;
	t->col = col;
	t->stride0 = stride0;
	t->stride1 = stride1;
	t->value_owner = 0;  //该指针是张量的主人

	t->grad = NULL;  //目前没有梯度的存在
	t->requires_grad = 0;  //目前不需要梯度传播
	t->parent_cnt = 0;
	t->parents = NULL;  //目前没有父节点
	t->backward_fn = NULL;  //目前没有记录该节点算子
	t->ctx = NULL;
	t->free_ctx = NULL;  //目前没有中间量
	t->mark = 0;  目前不需要记录到图中
	return t;
}

//确认该张量需要梯度时给梯度列表分配内存空间
void ensure_grad(tensor* t) {
	if (t->grad) return;
	int size = storage_size(t);
	t->grad = (float*)calloc(size, sizeof(float));  //给梯度数值列表分配内存空间
}
```

之前的向前传播算子也需要进行补充，以加法为例：

```
void add(tensor* c, tensor* a, tensor* b) {
	int size = a->row * a->col;
	for (int i = 0; i < size; i++) {
		c->value[i] = a->value[i] + b->value[i];
	}

	c->requires_grad = a->requires_grad || b->requires_grad;
    //只要有一个父节点需要梯度，那么该节点也就需要
	if (!c->requires_grad) return;

	c->parent_cnt = 2;
    //两个父节点
	c->parents = (tensor**)malloc(2 * sizeof(tensor*));
    //给父节点列表分配空间
	c->parents[0] = a;
	c->parents[1] = b;
    //记录两个父节点
	c->backward_fn = add_backward;
    //记录该节点用了什么算子
}
```

## 反向传播算子

我们要达成自动微分，就必须提前写好每一个反向传播算子，这样才可以让梯度传播能够自动进行，下面是三个例子：

1. 加法反向传播算子：

```
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
```

数学推导为：

$$
c=a+b\\[12pt]
\mathrm{grad}(a)=\frac{\partial{Loss}}{\partial{a}}=\frac{\partial{Loss}}{\partial{c}}\frac{\partial{c}}{\partial{a}}=\frac{\partial{Loss}}{\partial{c}}=\mathrm{grad}(c)\\[12pt]
\mathrm{grad}(b)=\frac{\partial{Loss}}{\partial{b}}=\frac{\partial{Loss}}{\partial{c}}\frac{\partial{c}}{\partial{b}}=\frac{\partial{Loss}}{\partial{c}}=\mathrm{grad}(c)
$$

相当于是直接把 $c$ 的梯度直接分别（按位）传给 $a$ 和 $b$ 。

2. 乘法反向传播算子：

```
void mul_backward(tensor* self) {
	tensor* a = self->parents[0];
	tensor* b = self->parents[1];
	if (a->requires_grad) {
		ensure_grad(a);
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
	if (b->requires_grad) {
		ensure_grad(b);
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
```

数学推导为：

$$
\mathbf C=\mathbf A \mathbf B\\[12pt]
c_{(i,j)}=\sum a_{(i,k)}b_{(k,j)}\\[12pt]
故:\mathrm{grad}(a_{(i,j)})=\frac{\partial{Loss}}{\partial{a_{(i,j)}}}=\sum \frac{\partial{Loss}}{\partial{c_{(i,k)}}}\frac{\partial{c_{(i,k)}}}{\partial{a_{(i,j)}}}=\sum \frac{\partial{Loss}}{\partial{c_{(i,k)}}}b_{(j,k)}=\sum \mathrm{grad}(c_{(i,k)})\times b_{(j,k)}
$$

由此已经可以计算出 $a$ 的梯度，同理可以算出 $b$ 的梯度。且整理可得结论：

$$
d\mathbf A=d\mathbf C B^T\\
d\mathbf B=A^Td\mathbf C
$$
3. 层归一化反向传播算子：
```
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
		mean /= d;  //计算均值
		float var = 0;
		for (int j = 0; j < d; j++) {
			float v = at(a, i, j) - mean;
			var += v * v;
		}
		var /= d;  //计算方差
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
```
数学推导为：
$$
首先回忆公式：\\[6pt]
\hat{x_i}=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\\[6pt]
y_i=\gamma\hat{x_i}+\beta\\[12pt]
则很简单的：\\[6pt]
\mathrm{grad}(\gamma_i)=\frac{\partial Loss}{\partial \gamma_i}=\sum\frac{\partial Loss}{\partial y_{(i,k)}}\frac{\partial y_{(i,k)}}{\partial \gamma_i}=\sum\frac{\partial Loss}{\partial y_{(i,k)}}\hat{x_k}=\sum\mathrm{grad}(y_{(i,k)})\times \hat{x_k}\\[12pt]
\mathrm{grad}(\beta_i)=\frac{\partial Loss}{\partial \beta_i}=\sum\frac{\partial Loss}{\partial y_{(i,k)}}\frac{\partial y_{(i,k)}}{\partial \beta_i}=\sum\frac{\partial Loss}{\partial y_{(i,k)}}=\sum\mathrm{grad}(y_{(i,k)})\\[20pt]
对于x的梯度会比较难算：\\[8pt]
\mathrm{grad}(x_i)=\frac{\partial Loss}{\partial x_i}=\\[6pt]
\frac{\partial Loss}{\partial \hat x_i}\frac{\partial \hat x_i}{\partial x_i}+\sum\frac{\partial Loss}{\partial \hat x}\frac{\partial \hat x}{\partial \mu}\frac{\partial \mu}{\partial x_i}+\sum\frac{\partial Loss}{\partial \hat x}\frac{\partial \hat x}{\partial \sqrt{\sigma^2+\epsilon}}\frac{\sqrt{\sigma^2+\epsilon}}{\partial x_i}\\[6pt]
=\frac{\partial Loss}{\partial y_i}\frac{\alpha}{\sqrt{\sigma^2+\epsilon}}+\sum\frac{\partial Loss}{\partial y}\frac{-\alpha}{n\sqrt{\sigma^2+\epsilon}}-\sum\frac{\partial Loss}{\partial y}\alpha\frac{x-\mu}{\sigma^2+\epsilon}\frac{x_i-\mu}{n\sqrt{\sigma^2+\epsilon}}\\[6pt]
=\mathrm{grad}(y_i)\frac{\alpha}{\sqrt{\sigma^2+\epsilon}}-\frac{\alpha}{n\sqrt{\sigma^2+\epsilon}}\sum\mathrm{grad}(y)-\frac{\alpha}{n\sqrt{\sigma^2+\epsilon}}\hat x_i\sum\mathrm{grad}(y)\hat x
$$
其中，在推导 $x$ 的梯度时，所有的视角都是在一行中看的，标记的 $i$ 指的是也是一行中的第 $i$ 个，是一个指定的位置，所有累加的下标 $k$ 已全部省略。
## 自动微分
### 构建拓扑图
自动微分依赖完整计算图，要求所有节点绑定反向传播算子。我们使用 DFS 深度优先搜索生成计算图拓扑序列，实现代码如下：
```
拓扑图收集：
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
```
其大致的原理就是，我们采用递归，直接把其中的每一条路走到底，然后返过来将节点填入列表，通过mark可以确保每一个节点一定会在第一次扫到就被记录，之后执行的时候沿着列表逆向进行，可以确保所有的梯度的传递到完整的链路上去，最终可以完成梯度更新。
<div  align="center">
<img src=assets/auto.png width=100% alt=auto>
</div>

### 内存管理
在梯度更新之前，我们不能所以释放张量，否则会导致反向传播断裂。同时，我们也不能释放权重张量，因此，我们需要维护和记录整个过程中的张量情况：
```
维护模型的参数：
struct model {
	tensor* src_E;  //源词表
	tensor* tgt_E;  //目标词表
	tensor* W_project;  //output投影矩阵（往往是目标词表的转置）
	layer* layers;  //编、解码器相关权重矩阵
	tensor* params[600];  //记录权重矩阵信息，避免释放张量时误伤
	int param_cnt;  //记录当前位置的游标
};
```
同时，由于在正向传播中有些算子是记录了一些中间张量来减轻反向传播时的计算量，因此我们需要记得释放它们，以交叉熵算子的中间量为例：
```
void free_cross_entropy_loss_ctx(void* p) {
	cross_entropy_loss_ctx* ctx = (cross_entropy_loss_ctx*)p;
	free_tensor(ctx->ex);
	free(ctx);
}
```
我们用通用的释放函数来释放所有的张量：
```
void free_tensor(tensor* t) {
	if (t->value_owner) {
		free(t->value);  //用于防止转置的歧义问题
	}
	free(t->grad);
	free(t->parents);  //malloc库函数
	if (t->ctx) {
		if (t->free_ctx) {
			t->free_ctx(t->ctx);  //如果有ctx中间量，则去释放
		}
		else {
			free(t->ctx);
		}
	}
	free(t);  //最终释放完毕
}
```
### 完整自动微分流程
```
反向传播循环：
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

反向传播+释放张量：
void train_backward(tensor* root, model* m) {
	tensor* order[2000];
	int cnt = 0;
	collect(root, order, &cnt);
	backward(order, cnt - 1);
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

梯度更新算子：
void sgd_update(tensor* t, float lr) {
	if (!t->requires_grad || !t->grad) return;
	for (int i = 0; i < t->row; i++) {
		for (int j = 0; j < t->col; j++) {
			set(t, i, j, at(t, i, j) - grad_at(t, i, j) * lr);
		}
	}
}

清空梯度与mark：
void zero_grad(tensor* t){
	if (t->grad) {
		int cnt = storage_size(t);
		for (int i = 0; i < cnt; i++) {
			t->grad[i] = 0;
		}
	}
	t->mark = 0;
}

梯度更新：
void update_model(model* m, float lr) {
	for (int i = 0; i < m->param_cnt; i++) {
		sgd_update(m->params[i], lr);
	}
}

void zero_model_grad(model* m) {
	for (int i = 0; i < m->param_cnt; i++) {
		zero_grad(m->params[i]);
	}
}
```
无需解释，调用前面写好的各类算子和函数即可。
## 训练
### 训练调度
将所有函数组合成一次单轮训练：
```
单轮训练：
float train_step(model* m, int* src_list, int len_src_list, int* tgt_list0, int* tgt_list1, int len_tgt_list, float lr) {
	zero_model_grad(m);  //初始化
	tensor* loss = train_forward(src_list, len_src_list, tgt_list0, tgt_list1, len_tgt_list, m);  //向前传播
	float loss_value = loss->value[0];
	train_backward(loss, m);  //向后传播
	update_model(m, lr);  //更新梯度
	return loss_value;  //输出本轮的Loss
}
```
### mini-batch
原来的梯度更新的时机是每一个句子反向传播后都开始更新，但是每次不同的句子可能导致梯度不断抖动，致使Loss停滞不减。而mini-batch就是讲连续的多个句子的梯度积累在一起后再统一更新。
### adam优化器
Adam优化器（Adaptive Moment Estimation）结合了动量法和RMSProp的思想，通过计算梯度的一阶矩（均值）和二阶矩（未中心化方差）来自适应地调整每个参数的学习率，它可以更好的在transformer这种大参数模型中优化参数的更新。
```
adam优化器：
void adam_update(model* m, float lr) {
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float eps = 0.00000001f;  //三个超参数
	m->adam_step += 1;  //用于记录当前有几个mini-batch
	float fix1 = 1.0f - powf(beta1, (float)m->adam_step);
	float fix2 = 1.0f - powf(beta2, (float)m->adam_step);
	for (int i = 0; i < m->param_cnt; i++) {
		tensor* t = m->params[i];
		if (!t->grad) continue;
		int size = storage_size(t);
		for (int j = 0; j < size; j++) {
			float g = t->grad[j];
			m->adam_m[i][j] = beta1 * m->adam_m[i][j] + (1.0f - beta1) * g;
			//一阶矩
			m->adam_v[i][j] = beta2 * m->adam_v[i][j] + (1.0f - beta2) * g * g;
			//二阶矩
			float now_m = m->adam_m[i][j] / fix1;
			float now_v = m->adam_v[i][j] / fix2;
			t->value[j] -= lr * now_m / (sqrtf(now_v) + eps);
		}
	}
}
```
### 算子性能优化：
以算子中最基础也最常见的矩阵乘法（mul）为例，我们最初的实现方法就是一个很基础的朴素三重循环，然而实际上，我们可以通过更高级的写法来大幅提升性能：
#### SIMD向量化：
```
#include <omp.h>
void mul(tensor* c, tensor* a, tensor* b) {
	//需要确保C矩阵为0
    int p = a->row;
    int q = a->col;
    int r = b->col;

    #pragma omp parallel for  //提示编译器多核处理
    for (int i = 0; i < p; i++) {

        for (int k = 0; k < q; k++) {
            float aik = a->value[i * q + k];  //交换j、k循环顺序

            #pragma omp simd  //提示编译器向量化
            for (int j = 0; j < r; j++) {
                c->value[i * r + j] += aik * b->value[k * r + j];
            }
        }
    }
}
```
其核心原理就是，将原有的 i-j-k 循环改写成 i-k-j 循环，即先固定 $A$ 中的每一个值，随后在 $B$ 中每行扫面并且与 $A$ 中定住的值相乘，并且累加到 $C$ 的对应位置。<br>
这样的核心好处就是可以大大节省cpu的内存读取开销，因为cpu本身就有一次读取8个连续内存的能力，但之前的写法cpu需要每次扫描 $B$ 中的非连续内存，而改写成向量化写法后cpu可以直接读取连续内存，这样可以极大的节省内存读取开销。同时由于每次的i、j循环都是独立的，因此可以直接调用多核并行计算，同样极大提升性能。<br>

#### 矩阵分块
```
#define BLOCK 64

void mul_blocked(tensor* c, tensor* a, tensor* b) {
	//需要确保C矩阵为0
    int p = a->row, q = a->col, r = b->col;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < p; i += BLOCK) {
        for (int kk = 0; kk < q; kk += BLOCK) {
            for (int j = 0; j < r; j += BLOCK) {
                //分块内执行 i-k-j 计算
                for (int ii = i; ii < i + BLOCK && ii < p; ii++) {
                    for (int k = kk; k < kk + BLOCK && k < q; k++) {
                        float aik = a->value[ii * q + k];
                        #pragma omp simd
                        for (int jj = j; jj < j + BLOCK && jj < r; jj++) {
                            c->value[ii * r + jj] += aik * b->value[k * r + jj];
                        }
                    }
                }
            }
        }
    }
}
```
该版本是SIMD的进一步进阶，即将一个大矩阵给分成多个小块，其解决的痛点就是如果目标矩阵是一个而很大的矩阵，那么 i-k-j 计算将会进行很长时间，但cpu的L1缓存只有32k，所以之后再想读取前面计算的值，就可能得去更低级的内存等地方找，效率远低于cpu自己的缓存。而小块的矩阵就不会有这个问题，始终保持最高的缓存交换速度，可以大幅提高cash命中率。

### KV Cache
KV Cache是Transformer标配的推理加速功能，在这里我们需要先把整个流程给梳理一遍：<br>

在训练的时候：
1. 编码器
2. 解码器

解码器输入的是一整个“答案”矩阵，通过掩码mask来防止模型提前看到后文信息。

而在生成的时候：
1. 编码器
2. 解码器
3. 解码器
4. 解码器<br>
...........

解码器输入的只是当前已生成的词矩阵，而且每一次完整的通过解码器只会生成出一个token。因此在这其中存在大量的重复计算，这也就是KV Cache的作用。

解码器的交叉注意力是不纳入KV Cache中的，虽然说它的KV矩阵也都可以缓存，但是由于它们其实都是静态的（因为编码不会改变），所以这里的缓存是显而易见的，故不纳入。<br>

实际上，解码器的自注意力是才是KV Cache的对象，其中，前面算出的KV矩阵将会被缓存下来，每次生成时只用算出新加的词的KV向量并且追加上去即可。而Q矩阵并不需要被缓存，这是因为我们此处根本不需要已生成词的Q向量，我们只需要最新一个词的Q向量即可（可以参考$Logits$矩阵的意义来看）。
<div  align="center">
<img src=assets/cache.png width=100% alt=cache>
</div>

### beam search 束搜索
1. 设置“束宽”B(Beam Width): 表示每一步保留的候选推理路径数量，比如 B=2，就保留概率 Top-2 的两个组合。

2. 按概率累积打分: 使用对数概率(logP)累加评分，避免数值太小无法比较。

3. 逐步扩展 & 筛选: 每一轮都从上一步的每个候选里继续“展开”，再从所有新组合中筛出 Top-B 个组合。

道理非常简单,就是说模型并不会直接选择概率最大的那个字（贪心算法），而是会探索多条路径，来找一个总体效果最好的生成语句。
<div  align="center">
<img src=assets/beam.png width=100% alt=beam>
</div>

### Temperature
结合上面的“束搜索”，在实际使用的大模型中，有一个可供用户调整的参数“Temperature”，它就是作用在logits的softmax过程中的：
$$
p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$
公式中的$T$就是Temperature参数，其中Temperature越小会导致概率分布越尖锐，从而使得模型的输出更加固定。<br>

当Temperature == 0时，模型的输出就会直接变为贪心算法，即直接取概率最大的那个数。很有意思的是，这并不代表模型每次的输出就一定相同了，事实是它们往往不会完全相同，这是因为：每一次生成包含了巨量的加减运算，而浮点数的加法是不满足结合律的，而推理的时候需要大量使用到gpu并行运算，任何一个结果的累计顺序有差别，都可能会导致最后的logits产生差别，从而导致输出不一致。
## 实机训练
（训练主要由ai独立完成）
### 训练环境：

| 项目 | 配置 |
| --- | - |
| 系统 | win 11 |
| CPU | 单核 Intel Core i7-13700HX |
| 内存 | 16G |

i7-13700HX有8个p性能核核8个e能效核，p核可以分成两个逻辑核，而e核可以分出一个逻辑核。这里的单核指的是一个逻辑核。具体的核在训练中由windows调度器分配，故实际性能差距可能很大。

### Transformer 配置

| 项目 | 数值 |
|---|---:|
| 编码器层数 | 6 |
| 解码器层数 | 6 |
| 模型维度 $dm$ | 144 |
| 注意力头数 $H$ | 6 |
| 每头维度 $dk$ | 24 |
| 前馈层维度 $dff$ | 576 |
| 参数张量数量 | 429 |
| 参数数量 | 5,002,128 个 float |

模型采用 Pre-LN 结构：即每个子层先 LayerNorm，再经过 Attention 或 FFN，最后与原输入相加。该修改用于解决原 Post-LN 六层网络中观察到的底层梯度消失。因为在第一次的训练尝试中发现梯度消失严重，故特此修改。

训练参数：
- 优化器为 Adam，`beta1=0.9`，`beta2=0.999`，`epsilon=1e-8`。
- 基础学习率为 `3e-4`，前 2,000 次更新做线性 warmup。

语料来源：
- Tatoeba / ManyThings 英中短句。
- OPUS TED2013 v1.1 英中平行语料。<br>

原始候选经过长度和质量过滤后，构建结果如下：

| 项目 | 数值 |
|---|---:|
| 长度过滤后的候选对 | 94,178 |
| 高频词表过滤后的候选对 | 63,175 |
| 最终去重样本数 | 62,157 |
| 训练集样本数 | 55,941 |
| 留出测试集样本数 | 6,216 |
| 英文词表大小 | 4,993 |
| 中文字符词表大小 | 2,746 |
| 源序列平均长度 | 8.91 token，含 BOS/EOS |
| 目标序列平均长度 | 11.38 token，含 BOS/EOS |
| 源序列最大长度 | 14 token |
| 目标序列最大长度 | 23 token |

<br>
预处理流程：<br>

1. 英文转小写，用正则提取单词及简单缩写。
2. 中文使用 OpenCC `t2s` 转为简体，只保留汉字字符。
3. 保留英语 2-12 词、中文 2-22 字的句对。
4. 保留中文长度/英文长度比在 0.8-4.0 的句对。
5. 统计高频英文词和中文字符，保留前 5,000 个英文词与前 3,000 个中文字符覆盖的句对。
6. 固定随机种子进行 reservoir sampling、去重和随机打乱。
7. 加入 `<pad>`、`<bos>`、`<eos>`、`<unk>` 四个特殊 token。
8. 按打乱后索引 `index % 10 == 0` 划分留出集，其余为训练集。

###  性能指标



| 项目 | 数值 |
|---|---:|
| Adam 更新次数 | 160,000 |
| batch size | 8 |
| 样本前后向次数 | 1,280,000 |
| 期望 epoch 数 | 22.88 |
| 总训练时间 | 58,040.11 秒，16.12 小时 |
| 每次更新时间 | 362.75 ms |
| 每个样本时间 | 45.34 ms |
| 更新吞吐 | 2.757 updates/s |

token/s 采用训练样本平均长度和均匀采样的期望值计算：

| 口径 | 数值 |
|---|---:|
| 训练期处理的源 token 总数 | 11,403,008 |
| 训练期处理的目标 token 总数 | 14,563,069 |
| 训练期源+目标 token 总数 | 25,966,077 |
| 目标 token 吞吐 | 250.9 token/s |
| 源+目标合计吞吐 | 447.4 token/s |

这里的 token/s 是端到端训练吞吐，包含逐样本张量分配、前向、自动微分、Adam 和释放计算图；不是单独矩阵乘法吞吐。

###  推理方式

推理使用自回归贪心解码：

1. 目标端输入从 `<bos>` 开始。
2. 运行编码器和解码器，取当前位置 logits 的 argmax。
3. 将预测 token 拼接到下一步 decoder 输入。
4. 预测到 `<eos>` 或达到 24 token 上限时停止。

### 最终评估结果

| 数据划分 | Cross Entropy | Perplexity | Teacher Token Accuracy | Teacher Exact Sentence | Greedy Exact Sentence |
|---|---:|---:|---:|---:|---:|
| 训练集，55,941 条 | 0.625597 | 1.869362 | 80.95% | 30.95% | 30.95% |
| 留出集，6,216 条 | 3.436609 | 31.081375 | 48.69% | 3.60% | 3.60% |
| 均匀随机基线 | 7.917901 | 2746 | 0.0364% | 不适用 | 不适用 |

### 测试集表现：

```text
English: i don't lie
Reference: 我不说谎
Greedy: 我不说谎

English: it's one of the great products of the human mind
Reference: 它是人类思维的其中一个伟大产物
Greedy: 是人类这是人类的一产品

English: why does it matter who cares
Reference: 为什么这一点很重要谁会在乎
Greedy: 这为什么管理学会变化
```
### 关于模型能力差距的分析——与谷歌论文中的模型相比
- 模型参数差别：<br>
论文中参数约65M，而本模型只有约5M参数。
- 训练数据差距：<br>
论文中有450万对“经典英德词典预料”，且经过更专业的清理，而本模型只有6.2万对粗糙的预料。
- token分词策略不同：<br>
一个token并不是严格的对应一个字，而很可能是一个常用的短语，也可能是一个“ex”、“ing”这样的前后缀。而原论文中有很专业的token分类器，是基于大规模的语料统计出来的合理token组合，而本模型直接是一个token对应一个字。

- 训练策略不同:<br>
本模型没有后续学习率衰减、真正的并行 batch、dropout、label smoothing、weight decay、梯度裁剪等小技巧。

- 生成解码方式不同：<br>
论文中使用beam search,而本模型直接使用的贪心算法



## END
本文介绍的这种最基本的transformer早已最简单的原始版本了，网上也有很多相关的教学，但是大部分都是用pytorch直接调库使用，故特此用C语言来实现它。<br>
当然，我写的代码在性能上必然有很多的局限性，文中最后提到的那些优化方法和一些组件也没有全部实装。并且整个代码中存在大量硬编码和定制设计的内容，并不是绝对的通用组件。自动微分的部分也没有考虑一些很复杂的循环结构。当然，经过我的实机测试，本文的代码可以进行正常训练，并且没有内存泄露，总的效果令人满意。<br>
而由于代码和文章写的时间很短，我也还不是专业人士，故可能存在各种不知名错误，请见谅！谢谢~
