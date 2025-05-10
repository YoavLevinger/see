from github_repo_complexity_evaluator_multiple_to_see import evaluate_repo_complexity

if __name__ == "__main__":
    owner = "monkeylearn"
    repo = "monkeylearn-python"

    result = evaluate_repo_complexity(owner, repo)
    print("\n🔧 Final Estimation Result:")
    for k, v in result.items():
        print(f"{k}: {v}")

    print('*************************************************************************')
    owner = "dotnet"
    repo = "android-samples"

    result = evaluate_repo_complexity(owner, repo)
    print("\n🔧 Final Estimation Result:")
    for k, v in result.items():
        print(f"{k}: {v}")


    print('*************************************************************************')
    owner = "jimdowling"
    repo = "id2210-vt14"

    result = evaluate_repo_complexity(owner, repo)
    print("\n🔧 Final Estimation Result:")
    for k, v in result.items():
        print(f"{k}: {v}")


    print('*************************************************************************')
    owner = "woorea"
    repo = "stacksherpa-js-old"

    result = evaluate_repo_complexity(owner, repo)
    print("\n🔧 Final Estimation Result:")
    for k, v in result.items():
        print(f"{k}: {v}")