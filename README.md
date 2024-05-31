# Interpreting Deep Visual Models with Sparse Autoencoders

This code allows you to reproduce all the results described in the final report, namely: train autoencoders on each Stable Diffusion block, calculate the activation frequencies of each hidden neuron and analyze their activations on various prompts.

# Scripts usage

## Generation with intervention
This script generates images after zeroing frequent or rear hidden neurons in each autoencoder separately. Also it generates corresponding images without intervention and with zeroing all neurons.

### Usage
```bash
python interpret_by_intervention.py [prompt] [save_dir] --ae_version [ae_version] --num_images_per_prompt [num_images]
```

### Arguments
* `prompt` - Prompt to be used for stable diffusion model.
* `save_dir` - Directory where images will be saved.
* `ae_version` - The version of autoencoder. Default is `15`.
* `num_images_per_prompt` - Number of generated images. Default is `3`.

### Example
```bash
python interpret_by_intervention.py "a table in a kitchen" "gen_imgs/" --num_images_per_prompt=2
```



## Analyzing neuron frequinces between two different prompts

This script allows to get the numbers of frequent neurons that were activated on one prompt `mult_threshold` times more often than on another and displays them on a histogram with frequencies of all neurons.

### Usage

```bash
python analyze_freqs.py [ae_number] [frequencies_path] [frequencies_to_compare_path] --subfolder [subfolder] --ae_version [ae_version] --mult_threshold [mult_threshold] --plot_savedir [plot_savedir]
```

### Arguments

* `ae_number` - The number of the autoencoder to analyze
* `frequencies_path` - The path of the file with frequencies of the 1st prompt.
* `frequencies_to_compare_path` - The path of the file with frequencies of the 2nd prompt.
* `subfolder` - The subfolder in the `ae_{ae_number}/` folder where `frequencies_path` and `frequencies_to_compare_path` are located.
* `ae_version` - The version of autoencoder. Default is `15`
* `mult_threshold` - Uses to find neurons which have `mult_threshold` times more frequencies with prompt 1 than with prompt 2. Default is `100`
* `plot_savedir` - The path where histogram will be saved.

### Example 
```bash
python analyze_freqs.py 5 freqs_green.npy freqs_table.npy --subfolder="ctrl_freqs/" --plot_savedir="analyzes/freqs_hist.png"
```