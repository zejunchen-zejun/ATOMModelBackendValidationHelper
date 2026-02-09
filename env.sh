alias caa='cd /home/zejchen@amd.com/plugin/ATOM/ATOMModelBackendValidationHelper'

# alias gg='git fetch && git checkout origin/zejun/plugin_for_atom_1223'
alias gg='git fetch && git checkout origin/zejun/plugin_for_atom_vllm_atom'
# alias ggg='git fetch && git checkout origin/zejun/model_impl'
# alias gbb='git checkout origin/zejun/plugin_for_atom_1223'

alias tt0='bash ./test.0.6B.tp1.no.custom.sh'
alias tt1='bash ./test.0.6B.tp1.sh'
alias tt3='bash ./test.235B.tp8.ep8.sh'
alias tt4='bash ./test.235B.tp8.ep8.original.sh'
alias tt5='RUNNING_IN_SUBPROCESS=1 pytest -vs test_atom.py --optional 2>&1 | tee log'

alias cc='bash ./curl.sh'
# alias kk='rm -rf ./*.pt'

# kill all
#             ps -eo pid,comm | grep -E '^( *[0-9]+ +(python3|tee|VLLM::))' | awk '{print $1}' | xargs kill -9

alias py-ep='python -c "import sys, importlib.metadata as im; d=im.distribution(sys.argv[1]); [print(f\"[{e.group}]\n  {e.name} = {e.value}\") for e in d.entry_points]"'

pip install --upgrade triton
pip install hf_transfer
pip install tblib
