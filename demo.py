import os
import subprocess

def run_experiment(config):
    tag = "Truth" if config['use_truth'] else f"Top{config['top_k']}"
    short_model = config['model_name'].split('/')[-1]
    
    log_name = f"{config['eval_dataset']}-{short_model}-{tag}"
    if config.get('attack_method'):
        log_name += f"-adv-{config['attack_method']}-{config['adv_per_query']}"
        
    log_dir = f"logs/{config['query_results_dir']}_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name}.txt")


    base_cmd = [
        "python3", "-u", "main.py",
        "--eval_model_code", config['eval_model_code'],
        "--eval_dataset",    config['eval_dataset'],
        "--model_name",      config['model_name'],
        "--adv_per_query",   str(config['adv_per_query']),
        "--gpu_id",          str(config['gpu_id']),
        "--seed",            str(config['seed']),
        "--defend", "--conflict" 
    ]
    
    print(f"[Running] {log_name}")
    subprocess.run(base_cmd)


if __name__ == "__main__":
    base_cfg = {
        'eval_model_code': "contriever",
        'query_results_dir': 'main',
        'use_truth': False,
        'top_k': 5,
        'gpu_id': 0,
        'attack_method': 'LM_targeted',
        'seed': 41,
        'note': None
    }

    datasets   = ['hotpotqa', 'nq', 'msmarco']
    models     = ["TroyDoesAI/Llama-3.1-8B-Instruct"]
    adv_counts = [5,4,3,2,1] #[poisoned rating 100% - 20%]

    for ds in datasets:
        for model in models:
            for num in adv_counts:
                run_experiment({
                    **base_cfg, 
                    'eval_dataset': ds, 
                    'model_name': model, 
                    'adv_per_query': num
                })