import yaml

# standard reward-bench grouping of the 23 subsets into 4 categories
RB_CATEGORY = {}
for category, subsets in {
    "Chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"],
    "Chat Hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst",
                  "llmbar-adver-GPTOut", "llmbar-adver-manual"],
    "Safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse", "xstest-should-respond",
               "donotanswer"],
    "Reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"],
}.items():
    for subset in subsets:
        assert subset not in RB_CATEGORY, f"{subset} listed twice"
        RB_CATEGORY[subset] = category
assert len(RB_CATEGORY) == 23


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)
