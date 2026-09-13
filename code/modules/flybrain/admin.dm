/*
 *  Админские команды: посадить муху за руль человека и посмотреть, что она делает.
 */

/// Создать тело для мухи. Если turf не задан — рядом с вызывающим.
/proc/flybrain_spawn_body(fly_name, turf/where)
	if(!where && usr)
		where = get_turf(usr)
	if(!where && SSjob && SSjob.fallback_landmark)
		where = get_turf(SSjob.fallback_landmark)   // спавн по topic, когда usr нет
	if(!where)
		return null
	var/mob/living/carbon/human/H = new(where)
	H.randomize_appearance()
	H.real_name = fly_name || "Drosophila melanogaster"
	H.name = H.real_name
	H.update_icons()
	return H

/client/proc/flybrain_spawn_fly()
	set category = "Fun"
	set name = "Fly Brain: посадить муху в человека"
	set desc = "Создаёт человека, за которого играет симуляция мозга дрозофилы"

	if(!holder)
		return
	if(!SSflybrain)
		to_chat(usr, "<span class='warning'>Подсистема flybrain не поднялась.</span>")
		return

	var/fly_name = input(usr, "Имя тела", "Fly Brain", "Drosophila melanogaster") as text|null
	if(isnull(fly_name))
		return

	var/mob/living/carbon/human/H = flybrain_spawn_body(fly_name, get_turf(mob))
	if(!H)
		to_chat(usr, "<span class='warning'>Не смог создать тело.</span>")
		return

	var/datum/fly_pilot/P = SSflybrain.attach(H)
	if(!P)
		to_chat(usr, "<span class='warning'>Не смог подключить мозг. Проверьте, \
что демон запущен и что в [FLYBRAIN_CONFIG_FILE] верный url.</span>")
		qdel(H)
		return

	message_admins("[key_name_admin(usr)] посадил муху ([P.id]) в [H].")
	to_chat(usr, "<span class='notice'>Муха [P.id] за рулём [H]. \
Транспорт: [SSflybrain.transport_name].</span>")

/client/proc/flybrain_attach_existing(mob/living/carbon/human/H in human_list)
	set category = "Fun"
	set name = "Fly Brain: отдать мухе этого человека"

	if(!holder || !istype(H))
		return
	if(H.client)
		if(alert(usr, "За [H] играет живой человек. Точно отдать тело мухе?",
			"Fly Brain", "Да", "Нет") != "Да")
			return
	var/datum/fly_pilot/P = SSflybrain.attach(H)
	if(P)
		message_admins("[key_name_admin(usr)] отдал [H] мухе ([P.id]).")

/client/proc/flybrain_detach()
	set category = "Fun"
	set name = "Fly Brain: отключить муху"

	if(!holder || !SSflybrain)
		return
	if(!length(SSflybrain.pilots))
		to_chat(usr, "Подключённых мух нет.")
		return
	var/list/choices = list()
	for(var/datum/fly_pilot/P in SSflybrain.pilots)
		choices["[P.id] ([P.body])"] = P
	var/pick = input(usr, "Кого отключаем?", "Fly Brain") as null|anything in choices
	if(!pick)
		return
	SSflybrain.detach(choices[pick])

/client/proc/flybrain_status()
	set category = "Debug"
	set name = "Fly Brain: статус"

	if(!holder || !SSflybrain)
		return
	var/list/out = list("<b>Fly Brain</b>",
		"включено: [SSflybrain.enabled]",
		"транспорт: [SSflybrain.transport_name]",
		"url: [SSflybrain.url]",
		"мух: [length(SSflybrain.pilots)]")
	for(var/datum/fly_pilot/P in SSflybrain.pilots)
		out += P.stat_line()
		if(P.last_error)
			out += "&nbsp;&nbsp;последняя ошибка: [P.last_error]"
		if(P.last_action && islist(P.last_action["dbg"]))
			var/list/dbg = P.last_action["dbg"]
			out += "&nbsp;&nbsp;курс [dbg["heading"]], поворот [dbg["turn"]]"
			if(islist(dbg["z"]))
				var/list/zs = list()
				for(var/k in dbg["z"])
					zs += "[k]=[dbg["z"][k]]"
				out += "&nbsp;&nbsp;z: [jointext(zs, " ")]"
	to_chat(usr, jointext(out, "<br>"))

/client/proc/flybrain_reload()
	set category = "Debug"
	set name = "Fly Brain: перечитать конфиг"

	if(!holder || !SSflybrain)
		return
	SSflybrain.load_config()
	SSflybrain.setup_transport()
	to_chat(usr, "flybrain: конфиг перечитан, транспорт [SSflybrain.transport_name]")
