# Here there are the configuration variables used through all the project

####### MODELS names and info ############################################ 
# - size (GB)
# (add more when needed)
MODELS = {
    "granite4:3b-h" : { "size_gb" : 4.4},
    "qwen2.5:3b" :    { "size_gb" : 2.0 },
    "llama3.2:3b"  :  { "size_gb" : 1.9 },
    "mistral:7b" :    { "size_gb" : 1.9 }
}

####### ARASAAC settings variable #########################################
ARASAAC_LANG = "en"
ARASAAC_BASE_URL = "https://api.arasaac.org/v1"
ARASAAC_STATIC_URL = "https://static.arasaac.org/pictograms"

####### LANGUAGE DEPENDENCIES #############################################
# To support other languages you have to add them
# TODO: other/better strategies? maybe more dynamic
DAYS = {"en" : ["monday", "tuesday", "thursday", ...]}
DAYS = DAYS[ARASAAC_LANG] # overwrites the dict with the list of current language
DAY_TIMES = {"en" : ["morning", "afternoon", "evening", "night"]}
DAY_TIMES = DAY_TIMES[ARASAAC_LANG] # overwrites the dict with the list of current language
SEASONS = {"en" : ["spring", "summer", "autumn", "winter"]}
SEASONS = SEASONS[ARASAAC_LANG] # overwrites the dict with the list of current language
