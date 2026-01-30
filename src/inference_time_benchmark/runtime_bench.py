# %% [markdown]
# # Benchmarking pLMs for runtime efficiency

# %%
# Common Imports
from transformers import AutoModel, MambaModel, MambaConfig
import torch
import json
import time

# %%
# Benchmarking Function

# Set trials
min_batch_size = 8
max_batch_size = 128
inc_batch_size = 8

min_sequence_length = 64
max_sequence_length = 512
inc_sequence_length = 64

iterations = 20

# Load to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(device)

model_results = []
def benchmark(model, model_name):
    curr_results = []
    print(f"Benchmarking {model_name} now")
    with torch.no_grad():
        print((' Using ' + device_name + ' ').center(80, '*'))
        print(' Start '.center(80, '*'))
        for sequence_length in range(min_sequence_length,max_sequence_length+1,inc_sequence_length):
            for batch_size in range(min_batch_size,max_batch_size+1,inc_batch_size):
                start = time.time()
                for i in range(iterations):
                    input_ids = torch.randint(1, 20, (batch_size,sequence_length)).to(device)
                    attention_mask = torch.ones_like(input_ids).to(device)
                    model(input_ids=input_ids, attention_mask=attention_mask)
                end = time.time()
                ms_per_protein = (end-start)/(iterations*batch_size)
                print('Sequence Length: %4d \t Batch Size: %4d \t Ms per protein %4.2f' %(sequence_length,batch_size,ms_per_protein))
                curr = {
                    "Sequence Length":   sequence_length,
                    "Batch Size":    batch_size,
                    "Ms per protein": ms_per_protein
                }
                model_results.append(curr)
                curr_results.append(curr)
                # save the results to json
                with open(f'./results/{model_name}_results.json', 'w') as f:
                    json.dump(curr_results, f)

            print(' Done '.center(80, '*'))
        print(' Finished '.center(80, '*'))


# # %% [markdown]
# # Benchmark ProtMamba

# # %%

# # Download from https://github.com/Bitbol-Lab/ProtMamba-ssm/releases/download/v1.0/ProtMamba_model-weights.zip and then unzip
# modelFolderPath = '/scratch/akeluska/ismb_submission/ProtMamba-Long-foundation'
# config = MambaConfig.from_pretrained(modelFolderPath)
# config.hidden_size = 512  # Match checkpoint dimensions

# mamba_model = MambaModel.from_pretrained(
#     modelFolderPath, 
#     ignore_mismatched_sizes=True,
#     config=config,
# ).to(device).eval()

# # Run benchmark
# results = benchmark(mamba_model, "ProtMamba")

# # %% [markdown]
# ## Benchmarking ProtT5-XLUniref-50

# # Original model to benchmark embedding creation runtime against.

# # %%
# # Model Import
# from transformers import T5EncoderModel

# # Load the model
# t5xl_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
# t5xl_model = t5xl_model.to(device)
# t5xl_model.eval()

# benchmark(t5xl_model, "ProtT5-XL")

# %% [markdown]
# Benchmark Ankh
# 
# Updated model from Elneggar Lab, follow-up to ProtTrans

# %%
import ankh

# Load base model
ankh_model = ankh.load_base_model()
ankh_model = ankh_model[0].to(device)
ankh_model.eval()

benchmark(ankh_model, "Ankh Base")

# %%
import ankh

# Load large model
ankh_model = ankh.load_large_model()
ankh_model = ankh_model[0].to(device)
ankh_model.eval()

benchmark(ankh_model, "Ankh Large")

# %% [markdown]
# Benchmark Lobster
# 
# Optimized pLMs

# %%
from lobster.model import LobsterPMLM
lobster_model = LobsterPMLM("asalam91/lobster_24M")
lobster_model.eval().to(device).to(torch.float32)

benchmark(lobster_model, "Lobster 24M")

# %%
lobster_150M_model = LobsterPMLM("asalam91/lobster_150M")
lobster_150M_model.eval().to(device).to(torch.float32)

benchmark(lobster_150M_model, "Lobster 150M")
