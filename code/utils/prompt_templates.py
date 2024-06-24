def create_prompt_subtitles(data):
    prompt = "En bruger har for nylig læst artikler: "
    for s in data["subtitles"]:
        s = s[:min(len(s), 60)]  # clip long subtitles
        prompt += f"{s}, "
    prompt += f"vil brugeren læse artiklen {data['subtitle']}? (ja/nej)\n"
    return prompt


def create_prompt_titles(data):
    prompt = "En bruger har for nylig læst artikler: "
    for t in data["titles"]:
        t = t[:min(len(t), 60)]  # clip long titles
        prompt += f"{t}, "
    prompt += f"vil brugeren læse artiklen {data['title']}? (ja/nej)\n"
    return prompt


def create_prompt_qa_fast(titles):
    prompt = "En bruger har for nylig læst artikler: "
    for t in titles:
        t = t[:min(len(t), 60)]  # clip long titles
        prompt += f"{t}, "
    prompt += "hvilken af følgende artikler er brugeren mest tilbøjelig til at klikke på?\n"
    return prompt


def create_prompt_diversity(data):
    titles, categories = data["titles"], data["categories"]
    assert len(titles) == len(categories)
    prompt = "En bruger har for nylig læst artikler: "
    for t, c in zip(titles, categories):
        t = t[:min(len(t), 60)]  # clip long titles
        prompt += f"({c}) {t}, "
    prompt += f"vil brugeren læse artiklen ({data['category']}) {data['title']}? (ja/nej)\n"
    return prompt


def create_prompt_w_publishtime(data):
    def get_time_diff(t1, t2):
        diff = str(t2-t1)  # nicely handled by datetime
        if diff[0] != "0":  # equal would indicate 0 days
            return " ".join(diff.split(" ")[:-1])  # k days, only look at days ignore hours
        return diff.split(" ")[-1:][0].split(":")[0] + " hours"  # k hours, here days is 0 so not mentioned

    # note that we use the difference as we dont think mT5 knows how far dates are apart
    prompt = "En bruger har for nylig læst artikler: "
    for t, p in zip(data["titles"], data["publish_times"]):
        t = t[:min(len(t), 60)]  # clip long titles
        prompt += f"({get_time_diff(p, data['publish_time'])}) {t}, "
    prompt += f"vil brugeren læse artiklen {data['title']}? (ja/nej)\n"
    return prompt
