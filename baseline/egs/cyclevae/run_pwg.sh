#! /bin/bash
gpu=0                      # gpu id
conf=default               # name of config 
mode=mixed                 # training mode, choices=["natural", "mixed"]
evalset=dv50_vcc2020_24kHz # evaluation set
resume=-1                  # number of iteration of resume model
iter=400000                # number of iteration of testing model
stage=                     # running stage (1-3)
                           # stage 1: PWG training
                           # stage 2: PWG decoding (analysis-synthesis)
                           # stage 3: PWG decoding (voice conversion)
. parse_options.sh || exit 1;

srcspks=("SEF1" "SEF2" "SEM1" "SEM2")
espks=("TEF1" "TEF2" "TEM1" "TEM2")
fspks=("TFF1" "TFM1")
gspks=("TGF1" "TGM1")
mspks=("TMF1" "TMM1")
tarspks=("${espks[@]}" "${fspks[@]}" "${gspks[@]}" "${mspks[@]}")
allspks=("${srcspks[@]}" "${espks[@]}" "${fspks[@]}" "${gspks[@]}" "${mspks[@]}")

# PWG training
if echo ${stage} | grep -q 1; then
    echo "PWG training."
    if (($resume > 0)); then
        python runpwg.py -g ${gpu} -c ${conf} -M ${mode} -R ${resume} -1
    else
        python runpwg.py -g ${gpu} -c ${conf} -M ${mode} -1
    fi
fi

# PWG decoding w/ acoustic features
if echo ${stage} | grep -q 2; then
    echo "PWG decoding."
    python runpwg.py -g ${gpu} -c ${conf} -M ${mode} \
    -I ${iter} -e ${evalset} -2
fi

# PWG decoding w/ speaker voice converted acoustic features
if echo ${stage} | grep -q 3; then
    for srcspk in ${srcspks[*]};
    do
        for tarspk in ${tarspks[*]};
        do
            echo "PWG decoding (${srcspk}-${tarspk})."
            python runpwg.py -g ${gpu} -c ${conf} -M ${mode} \
            -I ${iter} -s ${srcspk} -t ${tarspk} -e ${evalset} -3
        done
    done
fi