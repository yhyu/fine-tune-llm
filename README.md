# fine-tune-Llama2
Fine-tune [meta Llama2](https://ai.meta.com/llama/) with [QLoRA](https://arxiv.org/abs/2305.14314) approach.

## Objective
This repo is to demostrate how to fine-tune [meta Llama2](https://ai.meta.com/llama/) on your laptop.

## Prerequisites
Before you can download LLama2, you have to fill out this [form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) (remember to use the same email as you Huggingface account), and login Huggingface in advance.
```console
huggingface-cli login
```

## Inference
You can fine-tune LLama2 with your own data, or try the adapter [EdwardYu/llama-2-7b-MedQuAD](https://huggingface.co/EdwardYu/llama-2-7b-MedQuAD) pre-trained on the [MedQuAD](https://github.com/abachaa/MedQuAD) dataset.  
```python
base_model = "meta-llama/Llama-2-7b-chat-hf"
adapter = 'EdwardYu/llama-2-7b-MedQuAD'

tokenizer = AutoTokenizer.from_pretrained(adapter)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)
model = PeftModel.from_pretrained(model, adapter)

question = 'What are the side effects or risks of Glucagon?'
inputs = tokenizer(question, return_tensors="pt").to("cuda")
outputs = model.generate(inputs=inputs.input_ids, max_length=1024)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

To run model inference faster, you can load in 16-bits without 4-bit quantization.
```python
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

The alternative way to run the model inference without logging into Huggingface (skip the [Prerequisites](#prerequisites) section) is to use the merged version of the model [llama-2-7b-MedQuAD-merged](EdwardYu/llama-2-7b-MedQuAD-merged).
