/*
 *  /datum/fly_pilot — связка "тело в игре" <-> "мозг в питоне".
 *
 *  Цикл на каждый тик подсистемы:
 *      1. забрать ответ на прошлый запрос и исполнить команду
 *      2. собрать новый сенсорный пакет и отправить
 *
 *  Ответ приходит через тик — то есть муха живёт с задержкой в один
 *  сенсорный тик (200 мс по умолчанию). Для мухи это даже щедро:
 *  латентность её собственной зрительно-моторной петли ~30-60 мс,
 *  но в SS13 всё равно нет смысла дёргаться чаще, чем ходит персонаж.
 */

/datum/fly_pilot
	var/id
	var/mob/living/carbon/human/body
	var/datum/flybrain_transport/transport

	var/handle                      // хэндл текущего запроса
	var/sent_at = 0
	var/ticks = 0
	var/errors = 0
	var/last_error

	var/list/last_action
	var/last_state
	var/next_move_at = 0
	var/next_act_at = 0
	var/next_idle_per = 0       // чтобы хоботок не тянулся вхолостую каждый тик
	var/last_damage = -1
	var/bumped_recently = FALSE
	var/bump_dir = 0            // куда упёрлась: щетинки у мухи сомато­топичны
	var/sound_level = 0

	var/enabled = TRUE
	var/list/latency = list()       // последние замеры round-trip, в децисекундах

/datum/fly_pilot/New(mob/living/carbon/human/H, ident, datum/flybrain_transport/T)
	body = H
	id = ident || "fly_[REF(H)]"
	transport = T

/datum/fly_pilot/Destroy()
	body = null
	transport = null
	return ..()

/datum/fly_pilot/proc/fail(msg)
	errors++
	last_error = msg
	if(errors <= 3 || errors % 50 == 0)
		log_debug("flybrain [id]: [msg]")

/datum/fly_pilot/proc/tick()
	if(!enabled)
		return
	var/mob/living/carbon/human/H = body
	if(QDELETED(H) || !H.loc)
		SSflybrain.detach(src)
		return
	if(H.stat == DEAD)
		// мёртвая муха не летает; пилота отцепляем, тело остаётся
		SSflybrain.detach(src)
		return

	// --- 1. ответ на прошлый запрос -----------------------------------
	if(handle)
		var/res = transport.poll(src)
		if(isnull(res))
			if(world.time - sent_at > FLYBRAIN_TIMEOUT)
				handle = null
				fail("таймаут ответа")
			else
				return                     // ждём дальше, новый пакет не шлём
		else if(res == "")
			handle = null
		else
			latency += (world.time - sent_at)
			if(length(latency) > 20)
				latency.Cut(1, 2)
			var/list/act
			try
				act = json_decode(res)
			catch
				fail("не разобрал ответ: [copytext(res, 1, 160)]")
			if(islist(act))
				apply_action(act)

	// --- 2. новый сенсорный пакет -------------------------------------
	ticks++
	var/list/p = build_percept()
	if(!p)
		return
	var/body_text
	try
		body_text = json_encode(p)
	catch(var/exception/e)
		fail("json_encode: [e]")
		return
	sent_at = world.time
	if(!transport.send(src, body_text))
		handle = null

/datum/fly_pilot/proc/avg_latency()
	if(!length(latency))
		return 0
	var/s = 0
	for(var/v in latency)
		s += v
	return round(s / length(latency), 0.1)

/datum/fly_pilot/proc/stat_line()
	return "[id]: тиков [ticks], ошибок [errors], RTT [avg_latency()]дс, \
состояние [last_state || "?"]"
