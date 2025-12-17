
LLM2Vec is a simple recipe to convert decoder-only LLMs into text encoders. It consists of 3 simple steps: 1) enabling bidirectional attention, 2) training with masked next token prediction, and 3) unsupervised contrastive learning. The model can be further fine-tuned to achieve state-of-the-art performance.

<p align="center">
  <img src="https://github.com/McGill-NLP/llm2vec/assets/12207571/48efd48a-431b-4625-8e0f-248a442e3839" width="75%" alt="LLM2Vec_figure1"/>
</p>

## Training 
### MNTP training
To train the model with Masked Next Token Prediction (MNTP), you can use the `experiments/run_mntp.py` script. It is adapted from HuggingFace Masked Language Modeling (MLM) [script](https://github.com/huggingface/transformers/blob/51bcadc10a569847b93a30dbe3a077037ae63bad/examples/pytorch/language-modeling/run_mlm.py). To train the S-Llama-1.3B model with MNTP, run the following command:

```bash
python experiments/run_mntp.py train_configs/mntp/MetaLlama3.json
```

The Sheared-Llama-1.3B training configuration [file](train_configs/mntp/Sheared-Llama.json) contains all the training hyperparameters and configurations used in our paper. 
```json
{
    "model_name_or_path": "princeton-nlp/Sheared-LLaMA-1.3B",
    "dataset_name": "wikitext",
    "dataset_config_name": "wikitext-103-raw-v1",
    "mask_token_type": "blank",
    "data_collator_type": "default",
    "mlm_probability": 0.2,
    "lora_r": 16,
    "gradient_checkpointing": true,
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2"
    // ....
}
```




### Word-level experiments for this thesis

For the experiments in this thesis, we focus on word-level probing for Sheared LLaMA 1.3B and DeBERTa V3 Large on CoNLL-2003 (NER, POS) and UD English-EWT (XPOS/UPOS). All experiment configurations are defined as JSON files under `train_configs/word-task/` (for training) and `test_configs/word-task/` (for testing).

To run training locally with a JSON config, use:

```bash
python experiments/run_word_task.py train_configs/word-task/ShearedLlama-bi-mntp_ner.json
python experiments/run_word_task.py train_configs/word-task/ShearedLlama-bi-mntp_pos.json
python experiments/run_word_task_ewt.py train_configs/word-task/ShearedLlama-bi-mntp_pos_ewt.json
python experiments/run_word_task_unfrozen.py train_configs/word-task/ShearedLlama-bi-mntp-unfrozen-ner.json
python experiments/run_word_task_unfrozen.py train_configs/word-task/ShearedLlama-bi-mntp-unfrozen-pos.json
python experiments/run_word_task_unfrozen.py train_configs/word-task/ShearedLlama-unfrozen-mntp_pos_ewt.json

python experiments/run_word_task.py train_configs/word-task/DeBERTa-frozen-ner.json
python experiments/run_word_task.py train_configs/word-task/DeBERTa-frozen-pos.json
python experiments/run_word_task_ewt.py train_configs/word-task/DeBERTa-frozen-ewt.json
```

Each training config specifies the base model (`model_name_or_path`), optional MNTP checkpoint (`peft_addr`), dataset (`dataset_name`), task (`task`), training strategy (frozen vs full fine-tuning), and output directory. Corresponding test-time configurations in `test_configs/word-task/` can be run via:

```bash
python experiments/test_word_task.py --config_file test_configs/word-task/DeBERTa-frozen-ner.json
python experiments/test_word_task_deberta.py --config_file test_configs/word-task/DeBERTa-frozen-ewt.json
python experiments/test_word_task_ewt.py --config_file test_configs/word-task/ShearedLlama-bi-mntp_pos_ewt.json
```

On the HPC cluster used for this thesis, experiments are submitted via SLURM job scripts stored in the `jobs/` directory. Each `.job` file wraps a call to the appropriate Python experiment script in `llm2vec/experiments/` with a given JSON configuration, and manages resource allocation, logging, and checkpoint paths.


In order to train DeBERTa V3 Large on MaChAmp, refer to the documentation in https://github.com/machamp-nlp/machamp. Default parameters were used.

If you have any questions regarding reproducibility - feel free to open an issue.
