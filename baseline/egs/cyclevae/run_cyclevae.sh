#!/bin/bash
################################################################
#    SCRIPT FOR NON-PARALLEL VOICE CONVERSION WITH CycleVAE    #
################################################################

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# USER SETTINGS {{{
#######################################
#           STAGE SETTING             #
#######################################
# {{{
# 0: data list preparation step
# 1: feature extraction step
# 2: statistics calculation step
# 3: training step
# 4: decoding reconstruction step [for training neural vocoder/Paralel WaveGAN (PWG)]
# 5: decoding conversion step [for extracting converted features for synthesis w/ PWG]
# }}}
stage=012345
#stage=0 #data_list
#stage=1 #feet
#stage=2 #stat
#stage=3 #trn
#stage=4 #rec
#stage=5 #conv

#######################################
#          FEATURE SETTING            #
#######################################
# {{{
# shiftms: shift length in msec (default=5)
# fftl: fft length (default=1024)
# highpass_cutoff: highpass filter cutoff frequency (if 0, will not apply)
# mcep_dim: dimension of mel-cepstrum
# mcep_alpha: alpha value of mel-cepstrum
# n_jobs: number of parallel jobs for feature extraction
# }}}
shiftms=5
fftl=2048
highpass_cutoff=65
fs=24000
mcep_dim=49
mag=0.5
#n_jobs=1
#n_jobs=10
n_jobs=40
#n_jobs=45
#n_jobs=50
#n_jobs=60

#######################################
#          TRAINING SETTING           #
######################################
# {{{
# in_dim: dimension of input features
# lat_dim: dimension of latent features
# out_dim: dimension of output features
# n_cyc: number of cycle for CycleVAE, 0 means VAE
# hidden_layers: number of hidden layers for GRU
# hidden_units: number of hidden units for GRU
# kernel_size: number of kernel size for aux. convolution
# dilation_size: number of dilation size for aux. convolution
# lr: learning rate
# do_prob: dropout probability
# batch_size: batch frame size
# }}}

spks=(SEF1 SEF2 SEM1 SEM2 TFM1 TGM1 TMM1 TEF1 TEM1 TEF2 TEM2 TFF1 TGF1 TMF1)

data_name=vcc2020

# uv and log-f0 occupied the first two dimensions
# then codeap and mcep
## [uv,log-f0,codeap,mcep]
if [ $fs -eq 22050 ]; then
    stdim=4 # 2-dim code-aperiodicity
    wav_org_dir=wav_22.05kHz
    data_name=${data_name}_22.05kHz
    mcep_alpha=0.455 #22.05k ## frequency warping based on pysptk.util.mcepalpha
elif [ $fs -eq 24000 ]; then
    stdim=5 # 3-dim code-aperiodicity 
    wav_org_dir=wav_24kHz
    data_name=${data_name}_24kHz
    mcep_alpha=0.466 #24k
elif [ $fs -eq 48000 ]; then
    stdim=7 # 5-dim code-aperiodicity
    wav_org_dir=wav_48kHz
    data_name=${data_name}_48kHz
    mcep_alpha=0.554 #48k
elif [ $fs -eq 44100 ]; then
    stdim=7 # 5-dim code-aperiodicity
    wav_org_dir=wav_44.1kHz
    data_name=${data_name}_44.1kHz
    mcep_alpha=0.544 #44.1k
elif [ $fs -eq 8000 ]; then ## 8kHz not yet supported, WORLD fails to analyze and synthesize properly
    stdim=2
    wav_org_dir=wav_8kHz
    data_name=${data_name}_8kHz
    mcep_alpha=0.312 #8k
else
    stdim=3 #16k: 1-dim code-aperiodicity
    wav_org_dir=wav_16kHz
    data_name=${data_name}_16kHz
    mcep_alpha=0.41000000000000003 #16k
fi
## from WORLD: number of code-aperiodicities = min(15000,fs/2-3000)/3000
## [https://github.com/mmorise/World/blob/master/src/codec.cpp] line 212

out_dim=`expr ${mcep_dim} + 1`

trn=tr${out_dim}_${data_name}
dev=dv${out_dim}_${data_name}

in_dim=`expr ${stdim} + ${out_dim}`

lat_dim=32

n_cyc=2

hidden_layers=1

hidden_units=1024

kernel_size_enc=3
kernel_size_dec=3
dilation_size_enc=2
dilation_size_dec=2

lr=1e-4

batch_size_utt=5
batch_size_utt_eval=35

n_workers=2

pad_len=2300

batch_size=80

epoch_count=80

do_prob=0.5

mdl_name="cyclevae-mult-jnt-mix-scpost_laplace"

GPU_device=0
#GPU_device=1
#GPU_device=2

echo $mdl_name
echo $data_name
echo $GPU_device

#idx_resume=

### DECODING SETTINGS

#GPU_device_str="0"
#GPU_device_str="1"
#GPU_device_str="2"
#GPU_device_str="0,2"
#GPU_device_str="2,0"
#GPU_device_str="0,1"
#GPU_device_str="1,2"
#GPU_device_str="2,1"
#GPU_device_str="1,0"
GPU_device_str="0,1,2"
#GPU_device_str="1,2,0"
#GPU_device_str="1,0,2"
#GPU_device_str="2,1,0"

min_idx=78

n_gpus=1
n_gpus=2
n_gpus=3

# for reconstruction/list file of PWG training data
spks_trg_rec=(TEF1 TMF1 TEM1 TMM1 TFF1 TFM1 TGF1 TGM1 TEF2 TEM2 SEF1 SEM1 SEF2 SEM2)
spks_src_rec=(SEF1 SEM1 SEF2 SEM2)

# for conversion
spks_trg_dec=(TEF1 TMF1 TEM1 TMM1 TFF1 TFM1 TGF1 TGM1 TEF2 TEM2)
spks_src_dec=(SEF1 SEM1 SEF2 SEM2)

### END OF DECODING SETTINGS


# parse options
. parse_options.sh

# stop when error occured
set -e
# }}}


# STAGE 0 {{{
if [ `echo ${stage} | grep 0` ];then
    echo "###########################################################"
    echo "#                 DATA PREPARATION STEP                   #"
    echo "###########################################################"
    mkdir -p data/${trn}
    mkdir -p data/${dev}
    [ -e data/${trn}/wav.scp ] && rm data/${trn}/wav.scp
    [ -e data/${dev}/wav.scp ] && rm data/${dev}/wav.scp
    for spk in ${spks[@]};do
        if [ -n "$(echo $spk | sed -n 's/\(SE\)/\1/p')" ]; then
            echo vcc20_1 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 60 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 70 | tail -n 10 >> data/${dev}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(TE\)/\1/p')" ]; then
            echo vcc20_2 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 10 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 20 | tail -n 10 >> data/${dev}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 70 | tail -n 50 >> data/${trn}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(TF\)/\1/p')" ] \
            || [ -n "$(echo $spk | sed -n 's/\(TM\)/\1/p')" ]; then
            echo vcc20_3 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 70 | tail -n 60 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 70 | head -n 10 >> data/${dev}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/\(TG\)/\1/p')" ]; then
            echo vcc20_4 $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 70 | head -n 60 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 70 | tail -n 10 >> data/${dev}/wav.scp
        fi
    done
fi
# }}}


# STAGE 1 {{{
if [ `echo ${stage} | grep 1` ];then
    echo "###########################################################"
    echo "#               FEATURE EXTRACTION STEP                   #"
    echo "###########################################################"
    if true; then
    #if false; then
        # extract feat and wav_anasyn
        nj=0
        for set in ${trn} ${dev};do
            echo $set
            expdir=exp/feature_extract/${set}
            mkdir -p $expdir
            for spk in ${spks[@]}; do
                echo $spk
                minf0=`cat conf/${spk}.f0 | awk '{print $1}'`
                maxf0=`cat conf/${spk}.f0 | awk '{print $2}'`
                pow=`cat conf/${spk}.pow | awk '{print $1}'`
                echo $minf0 $maxf0 $pow
                scp=${expdir}/wav_${spk}.scp
                n_wavs=`cat data/${set}/wav.scp | grep "\/${spk}\/" | wc -l`
                echo $n_wavs
                if [ $n_wavs -gt 0 ]; then
                    cat data/${set}/wav.scp | grep "\/${spk}\/" > ${scp}
                    ${train_cmd} --num-threads ${n_jobs} ${expdir}/feature_extract_${spk}.log \
                        feature_extract.py \
                            --expdir exp/feature_extract \
                            --waveforms ${scp} \
                            --wavdir wav_anasyn/${set}/${spk} \
                            --wavfiltdir wav_filtered/${set}/${spk} \
                            --hdf5dir hdf5/${set}/${spk} \
                            --fs ${fs} \
                            --shiftms ${shiftms} \
                            --minf0 ${minf0} \
                            --maxf0 ${maxf0} \
                            --pow ${pow} \
                            --mcep_dim ${mcep_dim} \
                            --mcep_alpha ${mcep_alpha} \
                            --fftl ${fftl} \
                            --highpass_cutoff ${highpass_cutoff} \
                            --n_jobs ${n_jobs}
        
                    # check the number of feature files
                    n_feats=`find hdf5/${set}/${spk} -name "*.h5" | wc -l`
                    echo "${n_feats}/${n_wavs} files are successfully processed."

                    # update job counts
                    nj=$(( ${nj}+1  ))
                    if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
                        wait
                        nj=0
                    fi
                fi
            done
        done
    fi
    # make scp for feats
    set +e
    rm -f data/${trn}/feats_all.scp
    for set in ${trn} ${dev};do
        echo $set
        find hdf5/${set} -name "*.h5" | sort > tmp2
        find wav_filtered/${set} -name "*.wav" | sort > tmp3
        rm -f data/${set}/feats.scp data/${set}/wav_filtered.scp
        for spk in ${spks[@]}; do
            cat tmp2 | grep "\/${spk}\/" >> data/${set}/feats.scp
            cat tmp3 | grep "\/${spk}\/" >> data/${set}/wav_filtered.scp
            echo $set $spk
            cat tmp2 | grep "\/${spk}\/" >> data/${trn}/feats_all.scp
        done
        rm -f tmp2 tmp3
    done
    set -e
fi
# }}}


# STAGE 2 {{{
if [ `echo ${stage} | grep 2` ];then
    echo "###########################################################"
    echo "#            CALCULATE SPEAKER STATISTICS STEP            #"
    echo "###########################################################"
    expdir=exp/calculate_statistics
    if true; then
    #if false; then
        for spk in ${spks[@]};do
            echo $spk
            cat data/${trn}/feats.scp | grep \/${spk}\/ > data/${trn}/feats_spk-${spk}.scp
            cat data/${trn}/feats_spk-${spk}.scp > data/${trn}/feats_all_spk-${spk}.scp
            n_feats_dev=`cat data/${dev}/feats.scp | grep "\/${spk}\/" | wc -l`
            if [ $n_feats_dev -gt 0 ]; then
                cat data/${dev}/feats.scp | grep \/${spk}\/ >> data/${trn}/feats_all_spk-${spk}.scp
                cat data/${dev}/feats.scp | grep \/${spk}\/ > data/${dev}/feats_spk-${spk}.scp
            fi
            ${train_cmd} exp/calculate_statistics/calc_stats_${trn}_spk-${spk}.log \
                calc_stats.py \
                    --expdir ${expdir} \
                    --feats data/${trn}/feats_all_spk-${spk}.scp \
                    --stdim ${stdim} \
                    --stats data/${trn}/stats_spk-${spk}.h5
        done
        echo "speaker statistics are successfully calculated."
    fi
    echo "###########################################################"
    echo "#             CALCULATE JOINT STATISTICS STEP             #"
    echo "###########################################################"
    if true; then
    #if false; then
        ${train_cmd} exp/calculate_statistics/calc_stats_${trn}.log \
            calc_stats.py \
                --expdir ${expdir} \
                --feats data/${trn}/feats_all.scp \
                --stdim ${stdim} \
                --stats data/${trn}/stats_jnt.h5
        echo "joint statistics are successfully calculated."
    fi
fi
# }}}


if [ `echo ${stage} | grep 3` ] || [ `echo ${stage} | grep 4` ] || [ `echo ${stage} | grep 5` ]; then
set +e
spk_list="$(IFS="@"; echo "${spks[*]}")"
echo ${spk_list}

stats_list=()
feats_list=()
feats_eval_list=()
for spk in ${spks[@]};do
    stats_list+=(data/${trn}/stats_spk-${spk}.h5)
    feats_list+=(data/${trn}/feats_spk-${spk}.scp)
    n_feats_dev=`cat data/${dev}/feats.scp | grep "\/${spk}\/" | wc -l`
    if [ $n_feats_dev -eq 0 ]; then
        touch data/${dev}/feats_spk-${spk}.scp
    fi
    feats_eval_list+=(data/${dev}/feats_spk-${spk}.scp)
done

stats_list_list="$(IFS="@"; echo "${stats_list[*]}")"
feats_list_list="$(IFS="@"; echo "${feats_list[*]}")"
feats_list_eval_list="$(IFS="@"; echo "${feats_eval_list[*]}")"
set -e
fi


# STAGE 3 {{{
# set variables
setting=${mdl_name}_${data_name}_hl${hidden_layers}_hu${hidden_units}_ld${lat_dim}_kse${kernel_size_enc}_dse${dilation_size_enc}_ksd${kernel_size_dec}_dsd${dilation_size_dec}_cyc${n_cyc}_lr${lr}_bs${batch_size}_do${do_prob}_epoch${epoch_count}_bsu${batch_size_utt}_bsue${batch_size_utt_eval}_nwrk${n_workers}_pad${pad_len}

expdir=exp/tr${out_dim}_${setting}
if [ `echo ${stage} | grep 3` ];then
    echo "###############################################"
    echo "#               TRAINING STEP                 #"
    echo "###############################################"
    mkdir -p ${expdir} 
    echo $expdir
    if true; then
    #if false; then
        if [ $mdl_name == "cyclevae-mult-jnt-mix-scpost_laplace" ]; then
            #${cuda_cmd} ${expdir}/log/train_resume-${idx_resume}.log \
            ${cuda_cmd} ${expdir}/log/train.log \
                train_gru_cyclevae-mult-mix-scpost_laplace_batch.py \
                    --expdir ${expdir} \
                    --feats data/${trn}/feats.scp \
                    --feats_eval_list $feats_list_eval_list \
                    --stats_jnt data/${trn}/stats_jnt.h5 \
                    --spk_list $spk_list \
                    --in_dim ${in_dim} \
                    --stdim ${stdim} \
                    --lr ${lr} \
                    --hidden_units ${hidden_units} \
                    --batch_size ${batch_size} \
                    --batch_size_utt ${batch_size_utt} \
                    --batch_size_utt_eval ${batch_size_utt_eval} \
                    --stats_list $stats_list_list \
                    --out_dim ${out_dim} \
                    --lat_dim ${lat_dim} \
                    --n_cyc ${n_cyc} \
                    --kernel_size_enc ${kernel_size_enc} \
                    --dilation_size_enc ${dilation_size_enc} \
                    --kernel_size_dec ${kernel_size_dec} \
                    --dilation_size_dec ${dilation_size_dec} \
                    --epoch_count ${epoch_count} \
                    --hidden_layers ${hidden_layers} \
                    --do_prob ${do_prob} \
                    --n_workers ${n_workers} \
                    --GPU_device ${GPU_device} \
                    --pad_len ${pad_len}
                    #--resume ${expdir}/checkpoint-${idx_resume}.pkl \
        fi
    fi
fi
# }}}


# STAGE 4 {{{
if [ `echo ${stage} | grep 4` ];then
    echo $expdir $n_gpus $GPU_device $GPU_device_str
    config=${expdir}/model.conf
    model=${expdir}/checkpoint-${min_idx}.pkl
    for spk_trg in ${spks_trg_rec[@]};do
        if [ -n "$(echo $spk_trg | sed -n 's/\(T\)/\1/p')" ]; then
            if true; then
            #if false; then
                echo "#############################################################"
                echo "#                  DECODING RECONST. FEAT                   #"
                echo "#############################################################"
                echo $spk_trg $min_idx
                outdir=${expdir}/rec_${spk_trg}_${min_idx}
                mkdir -p $outdir
                feats_tr=data/${trn}/feats.scp
                feats_scp=${outdir}/feats_${spk_trg}.scp
                cat ${feats_tr} | grep "\/${spk_trg}\/" > ${feats_scp}
                feats_dv=data/${dev}/feats.scp
                cat ${feats_dv} | grep "\/${spk_trg}\/" >> ${feats_scp}
                ${cuda_cmd} ${expdir}/log/decode_rec_${spk_trg}_${min_idx}.log \
                    calc_rec_gru-cyclevae-mult-mix-scpost_laplace.py \
                        --feats ${feats_scp} \
                        --spk_list ${spk_list} \
                        --stats data/${trn}/stats_spk-${spk_trg}.h5 \
                        --stats_jnt data/${trn}/stats_jnt.h5 \
                        --mcep_alpha ${mcep_alpha} \
                        --outdir ${outdir} \
                        --model ${model} \
                        --config ${config} \
                        --GPU_device_str ${GPU_device_str} \
                        --n_gpus ${n_gpus}
                        #--GPU_device ${GPU_device} \
            fi
            if true; then
            #if false; then
                for spk_src in ${spks_src_rec[@]};do
                    if [ -n "$(echo $spk_src | sed -n 's/\(S\)/\1/p')" -a -n "$(echo $spk_trg | sed -n 's/\(T\)/\1/p')" ]; then
                        echo "#############################################################"
                        echo "#               DECODING CYCLIC RECONST. FEAT               #"
                        echo "#############################################################"
                        echo $spk_trg $spk_src $min_idx
                        outdir=${expdir}/cycrec_${spk_trg}-${spk_src}-${spk_trg}_${min_idx}
                        mkdir -p $outdir
                        feats_tr=data/${trn}/feats.scp
                        feats_dv=data/${dev}/feats.scp
                        feats_scp=${outdir}/feats_${spk_trg}.scp
                        cat ${feats_tr} | grep "\/${spk_trg}\/" > ${feats_scp}
                        cat ${feats_dv} | grep "\/${spk_trg}\/" >> ${feats_scp}

                        ${cuda_cmd} ${expdir}/log/decode_cycrec_${spk_trg}-${spk_src}-${spk_trg}_${min_idx}.log \
                            calc_cycrec_gru-cyclevae-mult-mix-scpost_laplace.py \
                                --feats ${feats_scp} \
                                --spk_list ${spk_list} \
                                --stats_src data/${trn}/stats_spk-${spk_src}.h5 \
                                --stats_trg data/${trn}/stats_spk-${spk_trg}.h5 \
                                --stats_jnt data/${trn}/stats_jnt.h5 \
                                --mcep_alpha ${mcep_alpha} \
                                --shiftms ${shiftms} \
                                --outdir ${outdir} \
                                --model ${model} \
                                --config ${config} \
                                --GPU_device_str ${GPU_device_str} \
                                --n_gpus ${n_gpus}
                                #--GPU_device ${GPU_device} \
                    fi
                done
            fi
        fi
    done
    feats=data/${trn}/feats.scp
    feats_eval=data/${dev}/feats.scp
    waveforms=data/${trn}/wav_filtered.scp
    waveforms_eval=data/${dev}/wav_filtered.scp
    feats_ftcyc_scp=data/${trn}/feats_ftcyc.scp
    feats_ftcyc_eval_scp=data/${dev}/feats_ftcyc.scp
    rm -f ${feats_ftcyc_scp} ${feats_ftcyc_eval_scp}
    waveforms_ftcyc_scp=data/${trn}/wav_filtered_ftcyc.scp
    waveforms_ftcyc_eval_scp=data/${dev}/wav_filtered_ftcyc.scp
    rm -f ${waveforms_ftcyc_scp} ${waveforms_ftcyc_eval_scp}
    for spk in ${spks_trg_rec[@]}; do
        echo $spk
        find hdf5/${trn}/${spk} -name "*.h5" | sort >> ${feats_ftcyc_scp}
        cat ${waveforms} | grep "\/${spk}\/" >> ${waveforms_ftcyc_scp}
        if [ -n "$(echo $spk | sed -n 's/\(T\)/\1/p')" ]; then
            find hdf5/${trn}/${spk}-${spk} -name "*.h5" | sort >> ${feats_ftcyc_scp}
            cat ${waveforms} | grep "\/${spk}\/" >> ${waveforms_ftcyc_scp}
        fi
        find hdf5/${dev}/${spk} -name "*.h5" | sort >> ${feats_ftcyc_eval_scp}
        cat ${waveforms_eval} | grep "\/${spk}\/" >> ${waveforms_ftcyc_eval_scp}
        if [ -n "$(echo $spk | sed -n 's/\(T\)/\1/p')" ]; then
            find hdf5/${dev}/${spk}-${spk} -name "*.h5" | sort >> ${feats_ftcyc_eval_scp}
            cat ${waveforms_eval} | grep "\/${spk}\/" >> ${waveforms_ftcyc_eval_scp}
            for spkcv in ${spks_src_rec[@]}; do
                if [ -n "$(echo $spkcv | sed -n 's/\(S\)/\1/p')" -a -n "$(echo $spk | sed -n 's/\(T\)/\1/p')" ]; then
                    echo tr $spk $spkcv
                    find hdf5/${trn}/${spk}-${spkcv}-${spk} -name "*.h5" | sort >> ${feats_ftcyc_scp}
                    cat ${waveforms} | grep "\/${spk}\/" >> ${waveforms_ftcyc_scp}
                fi
            done
            for spkcv in ${spks_src_rec[@]}; do
                if [ -n "$(echo $spkcv | sed -n 's/\(S\)/\1/p')" -a -n "$(echo $spk | sed -n 's/\(T\)/\1/p')" ]; then
                    echo dv $spk $spkcv
                    find hdf5/${dev}/${spk}-${spkcv}-${spk} -name "*.h5" | sort >> ${feats_ftcyc_eval_scp}
                    cat ${waveforms_eval} | grep "\/${spk}\/" >> ${waveforms_ftcyc_eval_scp}
                fi
            done
        fi
    done
fi
# }}}


if [ `echo ${stage} | grep 5` ];then
for spkr in ${spks_src_dec[@]};do
if [ -n "$(echo $spkr | sed -n 's/\(S\)/\1/p')" ]; then
for spk_trg in ${spks_trg_rec[@]};do
if [ $spkr != $spk_trg ]; then
echo $spkr $spk_trg $min_idx
# STAGE 5 {{{
    echo $expdir $n_gpus $GPU_device $GPU_device_str

    config=${expdir}/model.conf
    model=${expdir}/checkpoint-${min_idx}.pkl
    minf0=`cat conf/${spkr}.f0 | awk '{print $1}'`
    maxf0=`cat conf/${spkr}.f0 | awk '{print $2}'`
    pow=`cat conf/${spkr}.pow | awk '{print $1}'`

    echo "######################################################"
    echo "#                DECODING CONV. FEAT                 #"
    echo "######################################################"
    
    outdir=${expdir}/dev_${spkr}-${spk_trg}_${min_idx}
    mkdir -p ${outdir}
    
    waveforms=data/${dev}/wav.scp
    waveforms_scp=${outdir}/wavs_dv_${spkr}.scp
    cat $waveforms | grep "\/${spkr}\/" > ${waveforms_scp}
    h5outdir=hdf5/${dev}/${spkr}-${spk_trg}
    
    ${cuda_cmd} ${expdir}/log/decode_dev_${spkr}-${spk_trg}_${min_idx}.log \
        decode_gru-cyclevae-mult-mix-scpost_laplace.py \
            --waveforms ${waveforms_scp} \
            --outdir ${outdir} \
            --h5outdir ${h5outdir} \
            --model ${model} \
            --config ${config} \
            --spk_list ${spk_list} \
            --fs ${fs} \
            --mcep_alpha ${mcep_alpha} \
            --fftl ${fftl} \
            --shiftms ${shiftms} \
            --n_gpus ${n_gpus} \
            --minf0 ${minf0} \
            --maxf0 ${maxf0} \
            --pow ${pow} \
            --stats_src data/${trn}/stats_spk-${spkr}.h5 \
            --stats_trg data/${trn}/stats_spk-${spk_trg}.h5 \
            --GPU_device_str ${GPU_device_str} \
            --stats_jnt data/${trn}/stats_jnt.h5
            #--GPU_device ${GPU_device} \

    find ${h5outdir} -name "*.h5" | sort > data/${dev}/feats_cv_${spkr}-${spk_trg}.scp
fi
done
fi
done
fi
# }}}
