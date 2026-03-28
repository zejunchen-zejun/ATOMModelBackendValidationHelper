#!/bin/bash

curl -X POST "http://localhost:8000/start_profile"

curl -X POST "http://localhost:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "The capital of China is", "temperature": 0, "top_p": 1, "top_k": -1, "max_tokens": 64, "stream": false, "ignore_eos": false, "seed": 123
}'

curl -X POST "http://localhost:8000/stop_profile"
