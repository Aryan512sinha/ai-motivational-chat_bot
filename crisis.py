from typing import List


CRISIS_KEYWORDS: List[str] ={
    "suicidal","end","kill","quit","want to die","i quit","give up","hopeless",
    "no reason to live", "no one love me","this is end","ending it all"
}


SAFETY_MESSAGE=(
    "YOU ARE GOING TO TOUGH TIME."
    "YOU ARE NOT ALONE YOUR PARENTS AND FRIENDS ARE WITH YOU.\n"
    "YOUR LIFE MATTER AND YOU MATTER"
)

def contains_crisis_keywords(text: str)-> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)