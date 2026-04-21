import subprocess, time, json, urllib.request, os, signal

SWIFTLM_PATH = ".build/arm64-apple-macosx/release/SwiftLM"
MAIN_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
DRAFT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"

TESTS = [
    {"name": "Baseline (SSD Stream)", "flags": ["--model", MAIN_MODEL, "--stream-experts"]},
    {"name": "Speculative Decoding (SSD Stream + e4b draft)", "flags": ["--model", MAIN_MODEL, "--draft-model", DRAFT_MODEL, "--num-draft-tokens", "4", "--stream-experts"]},
]

def generate():
    prompt = "apple " * 384
    req = urllib.request.Request(
        "http://127.0.0.1:5422/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 60, "stream": True}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    tokens = 0
    ttft = None
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            for line in response:
                if line.startswith(b"data: ") and line != b"data: [DONE]\n":
                    if ttft is None: ttft = time.time() - start
                    tokens += 1
        tps = tokens / (time.time() - start - ttft) if tokens > 1 else 0
        return ttft, tps
    except Exception as e:
        print(f"Failed: {e}")
        return 0, 0

print("Profiling Speculative Decoding vs SSD Stream Baseline...\n")
for t in TESTS:
    print(f"=== {t['name']} ===")
    cmd = [SWIFTLM_PATH, "--port", "5422"] + t["flags"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for server
    for _ in range(60):
        try:
            if urllib.request.urlopen("http://127.0.0.1:5422/health").getcode() == 200: break
        except: time.sleep(1)
        
    ttft, tps = generate()
    print(f"  TTFT: {ttft:.2f}s | TPS: {tps:.2f} tok/s\n")
    
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    time.sleep(5)
