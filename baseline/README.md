# Voice Conversion with Cyclic Variational Autoencoder (CycleVAE)-based Spectral Modeling and Parallel WaveGAN (PWG)-based Waveform Modeling

----
## Requirements

Python 3.7

Cuda 10.0

----
## CycleVAE usage

Firstly, environment and necessary tools will be installed.

    $ cd tools
    $ make

Then, move to the experiment directory, and copy the waveform data of VCC 2020 to `wav_24kHz` folder.
    
    $ cd ../egs/cyclevae
    $ cp -r <path_to_vcc2020_data>/* wav_24kHz

So, the contents of `wav_24kHz` will be something like `wav_24kHz/SEF1`, `wav_24kHz/TFF1`, and so on.

Finally, all processes (stages) will be run from the script.
   
    $ bash run_cyclevae.sh

**NOTE:** 
- It takes ~2 days to train a CycleVAE model on TITAN V.


----
## CycleVAE script and folders

* **STAGE 0**: initialize list data of training and development sets

* **STAGE 1**: speech feature extraction (WORLD-based)

* **STAGE 2**: calculation of speech feature statistics

* **STAGE 3**: training of CycleVAE-based spectral modelling

* **STAGE 4**: decoding of reconstructed and cyclic-reconstructed spectral features (mel-cepstrum) for training of PWG-based neural vocoder

* **STAGE 5**: decoding of converted features for waveform generation with PWG


* `exp/tr50_<model_settings>`: contains CycleVAE model training experiments and decoding outputs with conventional vocoder

* `exp/tr50_<model_settings>/dev_<spk_src>-<spk_trg>_<model_index>`: contains output of converted waveforms (spk\_src into spk\_trg) with conventional vocoder (WORLD).

* an example of such trained model and generated features from `model_index = 78` can be accessed within [trained\_example](https://drive.google.com/open?id=117savGk8mZ7K_1BFEP0VQkIw_g83Vwh2) with the password of VCC 2020 database provided for the participants.

---
## List files for PWG training

* `data/tr50_vcc2020_24kHz/feats_ftcyc.scp`: list **h5** files of **training set** for PWG-based neural vocoder

* `data/tr50_vcc2020_24kHz/wav_filtered_ftcyc.scp`: list **waveform** (high-pass filtered) files of **training set** for PWG

* `data/dv50_vcc2020_24kHz/feats_ftcyc.scp`: list **h5** files of **development** set for PWG

* `data/dv50_vcc2020_24kHz/wav_filtered_ftcyc.scp`: list **waveform** (high-pass filtered) files of **development** set for PWG


---
## Files for PWG training

* For **PWG training**, **natural** speech features are located in **h5 path** `/feat_org_lf0`.

* On the other hand, **artificial** (reconstructed/cyclic-reconstructed) features are located in **h5 path** `/feat_recmcep_cycvae-<model_index>`.

* H5 files of **natural** speech features will be **located in** `hdf5/<tr_or_dv>50_vcc2020_24kHz/<spk>/` directory,

* and the h5 files of **artificial** features will be **located in** `hdf5/<tr_or_dv>50_vcc2020_24kHz/<spk_trg>-<spk_trg>/` or `hdf5/<tr_or_dv>50_vcc2020_24kHz/<spk_trg>-<spk_src>-<spk_trg>/`.


---
## Files for PWG decoding

* Finally, for **PWG decoding**, **converted** speech features are located in **h5 path** `/feat_cvmcep_cycvae-<model_index>`,

* where the h5 files of **converted** features will be **located in** `hdf5/dv50_vcc2020_24kHz/<spk_src>-<spk_trg>/` directory.


----
## PWG setup

Install ParallelWaveGAN.

    $ cd src
    $ pip install -e .


----
## PWG script and folders

* **STAGE 1**: PWG training (w/ WORLD-based features)

* **STAGE 2**: PWG decoding (w/ WORLD-based features)

* **STAGE 3**: PWG decoding (w/ speaker voice converted WORLD-based features)

* `conf_pwg/vcc2020.<config_version>.yml`: config file of PWG 

* `exp/parallel_wavegan_tr50_<model_settings>_<config_version>_<train_mode>`: the path to output trained PWG model 

* `exp/parallel_wavegan_tr50_<model_settings>_<config_version>_<train_mode>/wav/`: the path to output generated speech files


----
## PWG usage

* The repo provides two training sets (modes), which are training PWG with **natural** acoustic features (mode: **natural**) and training PWG with **artificial and natural** acoustic features (mode: **mixed**).

* The **mixed** PWG achieves higher speech quality than the **natural** PWG when the testing features are **converted** acoustic features. 

**gpuid**: (default: "0")  
**train_mode**: (default: "mixed", choices: ["mixed", "natural"])  
**config_version**: (default: "default")  

First, train PWG.  

    $ cd ../egs/cyclevae
    $ bash run_pwg.sh --gpu <gpuid> --stage 1 \
    --mode <train_mode> --conf <config_version>

Secondly, decode speech w/ natural or artificial features.
    
    $ bash run_pwg.sh --gpu <gpuid> --stage 2 \
    --mode <train_mode> --conf <config_version>

Finally, decode speech w/ each VC speaker pair.
    
    $ bash run_pwg.sh --gpu <gpuid> --stage 3 \
    --mode <train_mode> --conf <config_version>

**NOTE:** 
- It takes ~3 days to train a PWG model on TITAN V.
- The RTF of PWG generation is ~0.018 on TITAN V.
- The total training time of CycleVAE w/ PWG is ~5 days with single TITAN V.


---
## References

* CycleVAE [[paper]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2307.pdf) [[github]](https://github.com/patrickltobing/cyclevae-vc)

* PWG [[paper]](https://arxiv.org/abs/1910.11480) [[github]](https://github.com/kan-bayashi/ParallelWaveGAN)


---
## Authors

Development:   
Patrick Lumban Tobing @ Nagoya University ([@patrickltobing](https://github.com/patrickltobing))  
Yi-Chiao Wu @ Nagoya University ([@bigpon](https://github.com/bigpon))  

Advisor:  
Tomoki Toda @ Nagoya University

E-mail:  
`patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp`  
`yichiao.wu@g.sp.m.is.nagoya-u.ac.jp`  
`tomoki@icts.nagoya-u.ac.jp`
