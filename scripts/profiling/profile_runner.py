import argparse
import subprocess
import time
import urllib.request
import urllib.error
import json
import re
import sys
import os

CONFIGS = [
    {"name": "Dense/Vanilla", "flags": []},
    {"name": "SSD Stream", "flags": ["--stream-experts"]},
    {"name": "TurboQuant", "flags": ["--turbo-kv"]},
    {"name": "SSD + TurboQuant", "flags": ["--stream-experts", "--turbo-kv"]}
]

SWIFTLM_PATH = ".build/arm64-apple-macosx/release/SwiftLM"

def poll_health(port=5413, timeout=30):
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout:
        try:
            r = urllib.request.urlopen(url)
            if r.getcode() == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def make_request_stream(prompt_len, max_tokens, port=5413):
    # To prevent blowing up python memory when generating 100k prompts, build efficiently
    prompt = "apple " * int(prompt_len * 0.75)
    data = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True
    }).encode('utf-8')
    
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    ttft = None
    start = time.time()
    tokens = 0
    try:
        # Extreme context testing requires a very large socket timeout
        with urllib.request.urlopen(req, timeout=900) as response:
            for line in response:
                line = line.decode('utf-8').strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    if ttft is None:
                        ttft = time.time() - start
                    tokens += 1
            total_time = time.time() - start
            gen_time = total_time - ttft if ttft else 0
            tps = (tokens - 1) / gen_time if gen_time > 0 and tokens > 1 else 0
            return True, ttft, tps
    except Exception as e:
        print(f"Request failed: {e}")
        return False, 0, 0

def extract_base_memory(log_path):
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "Memory strategy: FULL GPU" in line:
                    m = re.search(r"\(([0-9.]+)GB model", line)
                    if m: return f"{m.group(1)} GB"
    except: pass
    return "N/A"

def extract_real_memory(log_path):
    try:
        with open(log_path, 'r') as f:
            log_data = f.read()
            m = re.findall(r"OS_RAM=([0-9.]+)", log_data)
            if m: return f"{m[-1]} GB"
    except: pass
    return "N/A"

def main():
    parser = argparse.ArgumentParser(description="Aegis-AI Physical Model Profiler")
    parser.add_argument("--model", required=True, help="Model ID (e.g. gemma-4-26b-a4b-it-4bit)")
    parser.add_argument("--out", default="./profiling_results.md", help="Output markdown file path")
    parser.add_argument("--contexts", default="512", help="Comma-separated list of context lengths to test (e.g. 512,40000,100000)")
    args = parser.parse_args()
    
    context_sizes = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    results = []
    
    subprocess.run(["killall", "SwiftLM"], stderr=subprocess.DEVNULL)
    
    for config in CONFIGS:
        print(f"\n==============================================")
        print(f"--- Profiling {args.model} [{config['name']}] ---")
        print(f"==============================================")
        
        model_path = f"/Users/simba/.aegis-ai/models/mlx_models/mlx-community/{args.model}"
        log_path = "./tmp/profile_server.log"
        cmd = [SWIFTLM_PATH, "--model", model_path] + config["flags"]
        
        with open(log_path, "w") as root_log:
            server_proc = subprocess.Popen(cmd, stdout=root_log, stderr=subprocess.STDOUT)
        
        if not poll_health(timeout=60):
            print("Server failed to start.")
            server_proc.terminate()
            continue
            
        static_mem = extract_base_memory(log_path)
        
        for ctx_size in context_sizes:
            print(f"\n>> Running {ctx_size}-token context test (max generation ~20)...")
            ok, ttft, tps = make_request_stream(prompt_len=ctx_size, max_tokens=20)
            
            real_mem = extract_real_memory(log_path)
            
            if ok:
                results.append({
                    "config": config["name"],
                    "context": ctx_size,
                    "ttft_20": f"{ttft:.2f}",
                    "tps_20": f"{tps:.2f}",
                    "static_mem": static_mem,
                    "real_mem": real_mem
                })
                print(f"Result [{config['name']} | Ctx: {ctx_size}]: TTFT={ttft:.2f}s TPS={tps:.2f} BaseRAM={static_mem} PhysRAM={real_mem}")
            else:
                print(f"Result [{config['name']} | Ctx: {ctx_size}]: FAILED / OOM")
                
        # Teardown after finishing all context sizes for this config
        server_proc.send_signal(subprocess.signal.SIGTERM)
        server_proc.wait(timeout=20)
        time.sleep(2) # Give OS memory manager a breather to reap active wires
        
    with open(args.out, "w") as f:
        f.write(f"### `{args.model}` - Extreme Context & Footprint Profile\n\n")
        f.write(f"Tested Context Lengths: {args.contexts}\n\n")
        f.write("| Configuration | Context Size | Time To First Token | Generation Speed | Theoretical Reservation | Peak OS Footprint (Active RAM) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['config']} | {r['context']} | {r['ttft_20']}s | {r['tps_20']} tok/s | {r['static_mem']} | {r['real_mem']} |\n")
            
    print(f"\nDone. Matrix saved to {args.out}")

if __name__ == "__main__":
    main()
