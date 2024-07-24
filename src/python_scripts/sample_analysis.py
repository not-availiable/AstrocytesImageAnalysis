import sys

def run_analysis(arg1, arg2):
    return f"Analysis complete with args: {arg1}, {arg2}"

if __name__ == "__main__":
    result = run_analysis(sys.argv[1], sys.argv[2])
    print(result)