**Data Toolformer**  
This is the implementation of the [Toolformer](https://arxiv.org/pdf/2302.04761.pdf) paper. We have used the following github repository which partially implemets this paper as reference for our implementation [https://github.com/conceptofmind/toolformer](https://github.com/conceptofmind/toolformer).  
  
**Experimental Setup**
- Model: Writer/Palmyra - 128M parameter model​  

- APIs implemented: Calendar, Calculator, WolframAlpha​  

- Datasets:​  

  - c4 – A colossal, cleaned version of CommonCrawl’s web crawl corpus​  

  - math_datset/arithmetic__add_or_sub_multiple – Math dataset with expressions​  

  - ChilleD/SVAMP - Math dataset that has word problems​  
   
**Architecture**  
<img width="1267" alt="toolformer" src="https://github.com/SJUWM/datatoolformer/assets/117421227/f60bed83-cc96-4b25-bf5d-e87bf77fb506">   
  
**Data Generation**  
Use the following command to generate the data  
```python data_generator.py --num_devices=x, --device_id=y```  
  
**How to finetune the model**  
Use the following command to finetune the palmyra-small(128 M params) model  
```python3 train_gptj_toolformer.py --model_name_or_path=Writer/palmyra-small --per_device_train_batch_size=1 --num_train_epochs 10 --save_strategy=epoch --output_dir=finetune_toolformer_v0 --dataset_name math_dataset --dataset_config_name arithmetic__add_sub_multiple  --tokenizer_name customToolformer --block_size 1024 --gradient_accumulation_steps 1 --do_train --do_eval --evaluation_strategy=epoch --logging_strategy=epoch --fp16 --overwrite_output_dir --adam_beta1=0.9 --adam_beta2=0.999 --weight_decay=2e-02 --learning_rate=1e-05 --warmup_steps=100 --per_device_eval_batch_size=1```  
