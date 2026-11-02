"""Send random prediction requests to the API so the dashboards have data."""
import argparse
import json
import random
import time
import urllib.error
import urllib.request

RANGES = {
    "sepal_length": (4.3, 7.9),
    "sepal_width": (2.0, 4.4),
    "petal_length": (1.0, 6.9),
    "petal_width": (0.1, 2.5),
}


def send(url, payload):
    request = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:30080")
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--rate", type=float, default=5, help="requests/sec")
    args = parser.parse_args()

    counts = {}
    end = time.time() + args.seconds
    while time.time() < end:
        payload = {k: round(random.uniform(*r), 1) for k, r in RANGES.items()}
        if random.random() < 0.05:
            payload["petal_width"] = -1  # invalid input, returns HTTP 422
        status = send(args.url, payload)
        counts[status] = counts.get(status, 0) + 1
        time.sleep(1 / args.rate)
    print("Responses by status code:", counts)


if __name__ == "__main__":
    main()
