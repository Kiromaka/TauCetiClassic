/*
 *  SSflybrain — подсистема, гоняющая всех подключённых мух.
 *
 *  Тикает раз в SS_WAIT_FLYBRAIN децисекунд. Никакой тяжёлой работы
 *  внутри мира не делает: собрать пакет и отдать его транспорту.
 */

SUBSYSTEM_DEF(flybrain)
	name = "Fly Brain"
	init_order = SS_INIT_FLYBRAIN
	wait = SS_WAIT_FLYBRAIN
	priority = SS_PRIORITY_DEFAULT
	runlevels = RUNLEVEL_GAME | RUNLEVEL_POSTGAME
	flags = SS_POST_FIRE_TIMING | SS_NO_INIT

	var/enabled = FALSE
	var/url = "http://127.0.0.1:5665"
	var/secret = ""
	var/transport_name = FLYBRAIN_TRANSPORT_RUSTG

	var/datum/flybrain_transport/transport
	var/list/pilots = list()          // список /datum/fly_pilot
	var/list/by_mob = list()          // mob -> pilot
	var/list/current_run = list()

/datum/controller/subsystem/flybrain/PreInit()
	load_config()
	// Регистрируем админские вербы без правки admin_verbs.dm.
	// Глобальные списки инициализируются раньше подсистем, так что это безопасно.
	global.admin_verbs_fun += list(
		/client/proc/flybrain_spawn_fly,
		/client/proc/flybrain_attach_existing,
		/client/proc/flybrain_detach,
	)
	global.admin_verbs_debug += list(
		/client/proc/flybrain_status,
		/client/proc/flybrain_reload,
	)

/datum/controller/subsystem/flybrain/proc/load_config()
	// Собственный конфиг, чтобы не править configuration.dm ядра.
	// Формат: KEY VALUE, по строке на параметр, # — комментарий.
	if(!fexists(FLYBRAIN_CONFIG_FILE))
		return
	for(var/line in splittext(file2text(FLYBRAIN_CONFIG_FILE), "\n"))
		line = trim(line)
		if(!length(line) || copytext(line, 1, 2) == "#")
			continue
		var/space = findtext(line, " ")
		var/key = lowertext(space ? copytext(line, 1, space) : line)
		var/value = space ? trim(copytext(line, space + 1)) : ""
		switch(key)
			if("enabled")     enabled = text2num(value) || (lowertext(value) == "true")
			if("url")         url = value
			if("secret")      secret = value
			if("transport")   transport_name = lowertext(value)

/datum/controller/subsystem/flybrain/proc/setup_transport(force_name)
	var/want = force_name || transport_name
	var/datum/flybrain_transport/T
	if(want == FLYBRAIN_TRANSPORT_FILE)
		T = new /datum/flybrain_transport/file(url, secret)
	else
		T = new /datum/flybrain_transport/rustg(url, secret)
		if(!T.available())
			log_debug("flybrain: rust_g не найден, откатываюсь на файловый транспорт")
			T = new /datum/flybrain_transport/file(url, secret)
	if(!T.available())
		log_debug("flybrain: транспорт [T.name] недоступен")
		return null
	transport = T
	transport_name = T.name
	return T

/datum/controller/subsystem/flybrain/proc/attach(mob/living/carbon/human/H, ident)
	if(!istype(H))
		return null
	if(by_mob[H])
		return by_mob[H]
	if(!transport && !setup_transport())
		return null
	var/datum/fly_pilot/P = new(H, ident, transport)
	pilots += P
	by_mob[H] = P
	enabled = TRUE
	log_debug("flybrain: [P.id] подключён к [H] ([transport_name])")
	return P

/datum/controller/subsystem/flybrain/proc/detach(datum/fly_pilot/P)
	if(!P)
		return
	pilots -= P
	current_run -= P
	if(P.body)
		by_mob -= P.body
	log_debug("flybrain: [P.id] отключён")
	qdel(P)

/datum/controller/subsystem/flybrain/proc/detach_mob(mob/M)
	detach(by_mob[M])

/datum/controller/subsystem/flybrain/fire(resumed = FALSE)
	if(!enabled || !length(pilots))
		return
	if(!resumed)
		current_run = pilots.Copy()
	while(length(current_run))
		var/datum/fly_pilot/P = current_run[length(current_run)]
		current_run.len--
		if(QDELETED(P))
			continue
		P.tick()
		if(MC_TICK_CHECK)
			return

/datum/controller/subsystem/flybrain/stat_entry(msg)
	..("мух:[length(pilots)] тр:[transport_name]")
