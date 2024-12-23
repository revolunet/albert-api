import os

BASE_URL = os.getenv("BASE_URL", "http://localhost:8080/v1")
EMBEDDINGS_MODEL_TYPE = "text-embeddings-inference"
LANGUAGE_MODEL_TYPE = "text-generation"
AUDIO_MODEL_TYPE = "automatic-speech-recognition"
RERANK_MODEL_TYPE = "text-classification"
INTERNET_COLLECTION_DISPLAY_ID = "internet"
PRIVATE_COLLECTION_TYPE = "private"
SUPPORTED_LANGUAGES = [
    "afrikaans",
    "albanian",
    "amharic",
    "arabic",
    "armenian",
    "assamese",
    "azerbaijani",
    "bashkir",
    "basque",
    "belarusian",
    "bengali",
    "bosnian",
    "breton",
    "bulgarian",
    "burmese",
    "cantonese",
    "castilian",
    "catalan",
    "chinese",
    "croatian",
    "czech",
    "danish",
    "dutch",
    "english",
    "estonian",
    "faroese",
    "finnish",
    "flemish",
    "french",
    "galician",
    "georgian",
    "german",
    "greek",
    "gujarati",
    "haitian",
    "haitian creole",
    "hausa",
    "hawaiian",
    "hebrew",
    "hindi",
    "hungarian",
    "icelandic",
    "indonesian",
    "italian",
    "japanese",
    "javanese",
    "kannada",
    "kazakh",
    "khmer",
    "korean",
    "lao",
    "latin",
    "latvian",
    "letzeburgesch",
    "lingala",
    "lithuanian",
    "luxembourgish",
    "macedonian",
    "malagasy",
    "malay",
    "malayalam",
    "maltese",
    "mandarin",
    "maori",
    "marathi",
    "moldavian",
    "moldovan",
    "mongolian",
    "myanmar",
    "nepali",
    "norwegian",
    "nynorsk",
    "occitan",
    "panjabi",
    "pashto",
    "persian",
    "polish",
    "portuguese",
    "punjabi",
    "pushto",
    "romanian",
    "russian",
    "sanskrit",
    "serbian",
    "shona",
    "sindhi",
    "sinhala",
    "sinhalese",
    "slovak",
    "slovenian",
    "somali",
    "spanish",
    "sundanese",
    "swahili",
    "swedish",
    "tagalog",
    "tajik",
    "tamil",
    "tatar",
    "telugu",
    "thai",
    "tibetan",
    "turkish",
    "turkmen",
    "ukrainian",
    "urdu",
    "uzbek",
    "valencian",
    "vietnamese",
    "welsh",
    "yiddish",
    "yoruba",
]
