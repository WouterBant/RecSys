def create_prompt(titles, subtitles, title, subtitle):
    # TODO maak vet mooie deense prompt hier
    # prompt = f"Given the following titles and subtitles of previously read articles:\n"
    # for i, (t, s) in enumerate(zip(titles, subtitles)):
    #     prompt += f"Article {i+1}:\nTitle: {t}\nSubtitle: {s}\n\n"
    # prompt += f"Is the user likely to click on an the following article:\n Title: {title}\n Subtitle {subtitle}? (yes/no)\n"
    # return prompt
    prompt = f"Givet følgende titler og undertekster fra tidligere læste artikler:\n"
    for i, (t, s) in enumerate(zip(titles, subtitles)):
        prompt += f"Artikel {i+1}:\nTitel: {t}\nUndertekst: {s}\n\n"
    prompt += f"Er brugeren sandsynligvis interesseret i at klikke på den følgende artikel:\n Titel: {title}\n Undertekst: {subtitle}? (ja/nej)\n"
    # of is ingen beter ipv nej
    return prompt