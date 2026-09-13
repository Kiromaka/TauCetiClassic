/*
 *  Необязательный обратный канал: Python -> сервер через world/Topic().
 *
 *  Нужен только для внеполосных команд (заспавнить муху, отцепить,
 *  перечитать конфиг). Основной цикл его не использует — моторная
 *  команда приезжает в теле HTTP-ответа на запрос из DM.
 *
 *  Чтобы канал заработал, в code/game/world.dm надо добавить три строки,
 *  см. INSTALL.md. Без этой правки всё остальное работает как есть.
 */

/world/proc/flybrain_topic(list/packet_data, addr)
	if(!SSflybrain)
		return "err=no_subsystem"
	if(SSflybrain.secret != "" && packet_data["secret"] != SSflybrain.secret)
		return "err=bad_secret"
	if(addr != "127.0.0.1" && addr != "::1")
		return "err=not_local"

	switch(packet_data["cmd"])
		if("ping")
			return "ok=1&pilots=[length(SSflybrain.pilots)]&transport=[SSflybrain.transport_name]"

		if("status")
			var/list/s = list("pilots" = length(SSflybrain.pilots),
				"transport" = SSflybrain.transport_name,
				"enabled" = SSflybrain.enabled)
			var/i = 0
			for(var/datum/fly_pilot/P in SSflybrain.pilots)
				s["fly[i]"] = P.stat_line()
				i++
			return list2params(s)

		if("spawn")
			var/mob/living/carbon/human/H = flybrain_spawn_body(packet_data["name"])
			if(!H)
				return "err=spawn_failed"
			var/datum/fly_pilot/P = SSflybrain.attach(H, packet_data["id"])
			return P ? "ok=1&id=[P.id]" : "err=attach_failed"

		if("detach")
			for(var/datum/fly_pilot/P in SSflybrain.pilots)
				if(P.id == packet_data["id"])
					SSflybrain.detach(P)
					return "ok=1"
			return "err=not_found"

		if("enable")
			SSflybrain.enabled = text2num(packet_data["value"]) ? TRUE : FALSE
			return "ok=1&enabled=[SSflybrain.enabled]"

		if("reload")
			SSflybrain.load_config()
			SSflybrain.setup_transport()
			return "ok=1&transport=[SSflybrain.transport_name]"

	return "err=unknown_cmd"
