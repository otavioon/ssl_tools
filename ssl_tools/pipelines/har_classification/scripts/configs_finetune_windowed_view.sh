declare -A MODELCONFIGS

####### Freeze

# CPC
MODELCONFIGS["cpc"]="cpc.py"
# GRU
MODELCONFIGS["gru1l128"]="gru_encoder.py"
MODELCONFIGS["gru2l128"]="gru_encoder.py"
# MLP
MODELCONFIGS["mlp1x64_fft"]="mlp.py"
MODELCONFIGS["mlp1x64"]="mlp.py"
MODELCONFIGS["mlp2x64_fft"]="mlp.py"
MODELCONFIGS["mlp2x64"]="mlp.py"
MODELCONFIGS["mlp3x64_fft"]="mlp.py"
MODELCONFIGS["mlp3x64"]="mlp.py"
# 1D conv
MODELCONFIGS["simple1Dconv"]="simple1Dconv_classifier.py"
MODELCONFIGS["simple1Dconv_fft"]="simple1Dconv_classifier.py"
# 2D conv
MODELCONFIGS["simple2Dconv"]="simple2Dconv_classifier.py"
MODELCONFIGS["simple2Dconv_fft"]="simple2Dconv_classifier.py"
# TFC
MODELCONFIGS["tfc_transformer"]="tfc.py"
# TFC Head
MODELCONFIGS["tfchead"]="tfc_head_classifier.py"
MODELCONFIGS["tfchead_fft"]="tfc_head_classifier.py"
# TNC Head
MODELCONFIGS["tnchead"]="tnc_head_classifier.py"
# TNC
MODELCONFIGS["tnc"]="tnc.py"

# Transformers
# MODELCONFIGS["transformer2l1h"]="transformer.py"
# MODELCONFIGS["transformer2l2h"]="transformer.py"

########  Update backbone

# CPC
MODELCONFIGS["cpc_non_freeze"]="cpc.py"
# GRU
MODELCONFIGS["gru1l128_non_freeze"]="gru_encoder.py"
MODELCONFIGS["gru2l128_non_freeze"]="gru_encoder.py"
# MLP
MODELCONFIGS["mlp1x64_fft_non_freeze"]="mlp.py"
MODELCONFIGS["mlp1x64_non_freeze"]="mlp.py"
MODELCONFIGS["mlp2x64_fft_non_freeze"]="mlp.py"
MODELCONFIGS["mlp2x64_non_freeze"]="mlp.py"
MODELCONFIGS["mlp3x64_fft_non_freeze"]="mlp.py"
MODELCONFIGS["mlp3x64_non_freeze"]="mlp.py"
# 1D conv
MODELCONFIGS["simple1Dconv_non_freeze"]="simple1Dconv_classifier.py"
MODELCONFIGS["simple1Dconv_fft_non_freeze"]="simple1Dconv_classifier.py"
# 2D conv
MODELCONFIGS["simple2Dconv_non_freeze"]="simple2Dconv_classifier.py"
MODELCONFIGS["simple2Dconv_fft_non_freeze"]="simple2Dconv_classifier.py"
# TFC
MODELCONFIGS["tfc_transformer_non_freeze"]="tfc.py"
# TFC Head
MODELCONFIGS["tfchead_non_freeze"]="tfc_head_classifier.py"
MODELCONFIGS["tfchead_fft_non_freeze"]="tfc_head_classifier.py"
# TNC Head
MODELCONFIGS["tnchead_non_freeze"]="tnc_head_classifier.py"
# TNC
MODELCONFIGS["tnc_non_freeze"]="tnc.py"

# Transformers
# MODELCONFIGS["transformer2l1h_non_freeze"]="transformer.py"
# MODELCONFIGS["transformer2l2h_non_freeze"]="transformer.py"

