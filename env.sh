alias cdd='/shared/amdgpu/home/zejun_chen_qle/plugin'

alias rrr='rocm-smi'
alias ppp='pkill -f python'

alias paa='pip list | grep aiter -i'
alias pvv='pip list | grep vllm -i'
alias pss='pip list | grep sgl -i'

alias ggg='git submodule update --init --recursive'
alias www='watch -n 1 'rocm-smi''

# alias caa='cd /home/zejchen@amd.com/plugin/ATOM/ATOMModelBackendValidationHelper'

# alias gg='git fetch && git checkout origin/zejun/plugin_for_atom_1223'
# alias gg='git fetch && git checkout origin/zejun/plugin_for_atom_vllm_atom'
# alias ggg='git fetch && git checkout origin/zejun/model_impl'
# alias gbb='git checkout origin/zejun/plugin_for_atom_1223'

# alias cc='bash ./curl.sh'
# alias kk='rm -rf ./*.pt'

# kill all
#             ps -eo pid,comm | grep -E '^( *[0-9]+ +(python3|tee|VLLM::))' | awk '{print $1}' | xargs kill -9

# alias py-ep='python -c "import sys, importlib.metadata as im; d=im.distribution(sys.argv[1]); [print(f\"[{e.group}]\n  {e.name} = {e.value}\") for e in d.entry_points]"'

# pip install --upgrade triton
# pip install hf_transfer
# pip install tblib
# pip install transformers==5.0.0

# # for kunlun bench
# pip install jsonlines prettytable oss2