from github_repo_complexity_evaluator_multiple_to_see import evaluate_repo_complexity

if __name__ == "__main__":
    owner = "CarlJi"
    repo = "RestfulAPITests"

    result = evaluate_repo_complexity(owner, repo)
    print("\n🔧 Final Estimation Result:")
    for k, v in result.items():
        print(f"{k}: {v}")

    print('*************************************************************************')
    owner = "service-mocker"
    repo = "service-mocker"

    result = evaluate_repo_complexity(owner, repo)
    print("\n🔧 Final Estimation Result:")
    for k, v in result.items():
        print(f"{k}: {v}")


    print('*************************************************************************')
    owner = "sayems"
    repo = "rest.api.test"

    result = evaluate_repo_complexity(owner, repo)
    print("\n🔧 Final Estimation Result:")
    for k, v in result.items():
        print(f"{k}: {v}")


    print('*************************************************************************')
    owner = "maccman"
    repo = "abba"

    result = evaluate_repo_complexity(owner, repo)
    print("\n🔧 Final Estimation Result:")
    for k, v in result.items():
        print(f"{k}: {v}")

    print('*************************************************************************')
    owner = "ant4g0nist"
    repo = "Susanoo"

    result = evaluate_repo_complexity(owner, repo)
    print("\n🔧 Final Estimation Result:")
    for k, v in result.items():
        print(f"{k}: {v}")