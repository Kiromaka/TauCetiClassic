var/global/list/ai_names = file2list("static/names/ai.txt")
var/global/list/wizard_first = file2list("static/names/wizardfirst.txt")
var/global/list/wizard_second = file2list("static/names/wizardsecond.txt")
var/global/list/ninja_titles = file2list("static/names/ninjatitle.txt")
var/global/list/ninja_names = file2list("static/names/ninjaname.txt")
var/global/list/commando_names = file2list("static/names/death_commando.txt")
var/global/list/first_names_male = file2list("static/names/first_male.txt")
var/global/list/first_names_female = file2list("static/names/first_female.txt")
var/global/list/last_names = file2list("static/names/last.txt")
var/global/list/clown_names = file2list("static/names/clown.txt")
var/global/list/mime_names = file2list("static/names/mime.txt")
var/global/list/pirate_first = file2list("static/names/piratefirst.txt")
var/global/list/pirate_second = file2list("static/names/piratesecond.txt")
var/global/list/moth_first = file2list("static/names/moth_first.txt")
var/global/list/moth_second = file2list("static/names/moth_second.txt")
var/global/list/serpentid_names = file2list("static/names/serpentid.txt")
var/global/list/tajaran_male_first = file2list("static/names/tajaran_male_first.txt")

// Traitors key-words
var/global/list/rus_nouns
var/global/list/rus_adjectives
var/global/list/rus_verbs
var/global/list/rus_occupations
var/global/list/rus_bays
var/global/list/rus_local_terms

//loaded on startup because of "
//would include in rsc if ' was used

var/global/list/forbidden_names = list("space","floor","wall","r-wall","monkey","unknown","inactive ai","plating","unknown","arrivals alert system")
