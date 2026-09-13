/*
 *  Исполнение моторной команды.
 *
 *  Питон присылает не "нажми W", а результат работы нисходящих нейронов:
 *  куда смотреть, идти ли вперёд/назад, бежать ли, и опционально
 *  поведенческий акт (груминг, угроза, еда, эскейп).
 */

/datum/fly_pilot/proc/apply_action(list/act)
	var/mob/living/carbon/human/H = body
	if(!istype(H) || H.stat == DEAD || !H.loc)
		return

	last_action = act
	last_state = islist(act["dbg"]) ? act["dbg"]["state"] : null

	// --- направление --------------------------------------------------
	var/want_dir = text2num("[act["dir"]]")
	if(want_dir in list(NORTH, SOUTH, EAST, WEST))
		if(H.dir != want_dir)
			H.set_dir(want_dir)

	// --- походка ------------------------------------------------------
	var/run = text2num("[act["run"]]")
	H.m_intent = run ? MOVE_INTENT_RUN : MOVE_INTENT_WALK

	// --- шаг ----------------------------------------------------------
	var/move = text2num("[act["move"]]")
	if(move && world.time >= next_move_at)
		var/dir_to_go = move > 0 ? H.dir : turn(H.dir, 180)
		var/turf/before = get_turf(H)
		step(H, dir_to_go)
		if(get_turf(H) == before)
			bumped_recently = TRUE      // упёрлись — это тактильный сигнал
			bump_dir = dir_to_go        // и упёрлись ИМЕННО с этой стороны
		next_move_at = world.time + max(1, 2 + H.movement_delay())

	// --- поведенческий акт --------------------------------------------
	var/what = act["act"]
	if(what && world.time >= next_act_at)
		next_act_at = world.time + 8
		switch(what)
			if("groom")
				do_groom(H, act["part"])
			if("threat")
				H.me_emote("резко разводит руки в стороны и замирает.")
			if("escape")
				H.me_emote("дёргается всем телом и бросается прочь.")
			if("eat")
				do_feed(H)
			if("drink")
				do_drink(H)
			if("climb")
				do_climb(H)
			if("halt")
				H.me_emote("замирает.")

	if(act["say"])
		H.say("[act["say"]]")

/// Пищевой акт. Три стадии, ровно как у мухи: попробовала лапками —
/// подобрала — вытянула хоботок и ест. Каждый вызов делает ОДНУ стадию,
/// поэтому со стороны это выглядит последовательностью, а не рывком.
/datum/fly_pilot/proc/do_feed(mob/living/carbon/human/H)
	if(H.stat != CONSCIOUS)
		return

	// Стадия 3: еда уже в руках — едим.
	var/obj/item/weapon/reagent_containers/food/held = null
	for(var/obj/item/I in list(H.get_active_hand(), H.l_hand, H.r_hand))
		if(istype(I, /obj/item/weapon/reagent_containers/food))
			held = I
			break
	if(held)
		H.me_emote("вытягивает хоботок к [held.name] и ест.")
		held.attack(H, H)
		return

	// Стадия 2: еда под ногами или в шаге — подбираем. Муха не умеет
	// "взять предмет" сама, поэтому это делает пилот. Без этого она
	// бесконечно тянется к лежащей на полу булке, и ничего не происходит.
	var/obj/item/weapon/reagent_containers/food/target = null
	for(var/obj/item/weapon/reagent_containers/food/F in get_turf(H))
		target = F
		break
	if(!target)
		for(var/turf/N in orange(1, H))
			for(var/obj/item/weapon/reagent_containers/food/F in N)
				target = F
				break
			if(target)
				break
	if(target)
		if(H.get_active_hand())
			H.drop_item()
		if(H.put_in_hands(target))
			H.me_emote("подхватывает [target.name].")
		else
			// Руки заняты намертво — едим прямо с пола, лишь бы не встать
			// в ступор у лежащей еды.
			H.me_emote("приникает к [target.name].")
			target.attack(H, H)
		return

	// Стадия 1: еды рядом нет. Вытягивание хоботка вхолостую — реальное
	// поведение голодной мухи, но не каждые полсекунды: так это выглядит
	// как поломка, а не как поведение.
	if(world.time >= next_idle_per)
		next_idle_per = world.time + 150
		H.me_emote("вытягивает губы трубочкой.")

// Громкие звуки рядом — вход в джонстонов орган.
// Столкновения ловим не через Bump(), а по факту неудавшегося шага
// в apply_action(): так модуль не переопределяет ничего в ядре.
/datum/fly_pilot/proc/hear_noise(volume = 1)
	sound_level = clamp(max(sound_level, volume), 0, 1)

/// Чистка. Щетинки у мухи сомато­топичны: раздражение конкретного места
/// запускает чистку ИМЕННО ЭТОГО места. Какую часть чистить, решает питон
/// по направлению касания, здесь только эмоут.
/datum/fly_pilot/proc/do_groom(mob/living/carbon/human/H, part)
	switch("[part]")
		if("голова")
			H.me_emote("торопливо протирает лапками голову и глаза.")
		if("левый бок")
			H.me_emote("заводит руки за левый бок и чистится.")
		if("правый бок")
			H.me_emote("заводит руки за правый бок и чистится.")
		if("брюшко")
			H.me_emote("обчищает себя снизу, будто задними лапками.")
		else
			H.me_emote("чистит лапками лицо.")

/// Питьё: вода или спирт. Стадии те же, что у еды — в руках пьём, рядом
/// подбираем. Дрозофила летит на брожение, так что бар для неё это цветник.
/datum/fly_pilot/proc/do_drink(mob/living/carbon/human/H)
	if(H.stat != CONSCIOUS)
		return

	var/obj/item/weapon/reagent_containers/held = null
	for(var/obj/item/I in list(H.get_active_hand(), H.l_hand, H.r_hand))
		if(!istype(I, /obj/item/weapon/reagent_containers))
			continue
		var/obj/item/weapon/reagent_containers/RC = I
		if(flybrain_has_booze(RC) || flybrain_has_water(RC))
			held = RC
			break
	if(held)
		if(flybrain_has_booze(held))
			H.me_emote("жадно присасывается к [held.name].")
		else
			H.me_emote("вытягивает хоботок в [held.name] и пьёт.")
		held.attack(H, H)
		return

	// Подобрать стакан рядом.
	var/obj/item/weapon/reagent_containers/target = null
	for(var/turf/N in view(1, H))
		for(var/obj/item/weapon/reagent_containers/RC in N)
			if(flybrain_has_booze(RC) || flybrain_has_water(RC))
				target = RC
				break
		if(target)
			break
	if(target)
		if(H.get_active_hand())
			H.drop_item()
		if(H.put_in_hands(target))
			H.me_emote("хватает [target.name].")
		else
			H.me_emote("приникает к [target.name].")
			target.attack(H, H)
		return

	// Воды в посуде нет — пьём с пола. Муха так и делает.
	var/turf/T = get_turf(H)
	if(flybrain_wet_turf(T))
		H.me_emote("припадает к полу и пьёт с мокрого пола.")

/// Отрицательный геотаксис: муха упорно ползёт вверх. В игре это значит
/// «залезть на стол». Имя процедуры подъёма в ветках разное, поэтому
/// пробуем по очереди и молча сдаёмся, если ни одной нет.
/datum/fly_pilot/proc/do_climb(mob/living/carbon/human/H)
	if(H.stat != CONSCIOUS)
		return
	var/obj/structure/target = null
	for(var/turf/N in view(1, H))
		for(var/obj/structure/S in N)
			if(flybrain_climbable(S))
				target = S
				break
		if(target)
			break
	if(!target)
		return
	H.me_emote("карабкается на [target.name].")
	if(hascall(target, "do_climb"))
		call(target, "do_climb")(H)
	else if(hascall(target, "climb_structure"))
		call(target, "climb_structure")(H)
	else
		// Ни одной знакомой процедуры подъёма нет: просто шагаем на клетку,
		// если она проходима. Ничего в ядре при этом не переопределяется.
		step_towards(H, target)
