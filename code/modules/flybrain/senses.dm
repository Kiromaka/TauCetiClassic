/*
 *  Сборка сенсорного пакета: что муха видит, чует и чувствует телом.
 *
 *  Поле зрения разворачивается в систему координат мухи прямо здесь,
 *  чтобы питоновская сторона ничего не знала про BYOND-овые dir.
 */

/datum/fly_pilot/proc/build_percept()
	var/mob/living/carbon/human/H = body
	if(!istype(H))
		return null

	var/turf/origin = get_turf(H)
	if(!origin)
		return null

	// --- поле зрения -------------------------------------------------
	var/list/light = new /list(FLYBRAIN_VIEW_CELLS)
	var/list/solid = new /list(FLYBRAIN_VIEW_CELLS)
	var/list/mobs  = new /list(FLYBRAIN_VIEW_CELLS)
	var/list/items = new /list(FLYBRAIN_VIEW_CELLS)
	for(var/i in 1 to FLYBRAIN_VIEW_CELLS)
		light[i] = 0
		solid[i] = 1        // невидимое считаем стеной: муха туда не полетит
		mobs[i]  = 0
		items[i] = 0

	var/facing = H.dir
	var/list/seen = list()
	for(var/turf/T in view(FLYBRAIN_VIEW_RADIUS, H))
		seen[T] = TRUE

	var/food_best = 0
	var/danger_best = 0
	// Направление на ближайшее живое существо, в системе координат мухи.
	// Нужно, чтобы эскейп уводил ОТ обидчика, а не просто разворачивал
	// на 180 градусов от текущего курса.
	var/mob/living/threat = null
	var/threat_dist = 999
	// Ближайшая еда — та же история, что с угрозой: без направления муха
	// "идёт к еде" по текущему курсу и мимо неё же и проходит.
	var/obj/item/food_src = null
	var/food_dist = 999
	// Выпивка, вода и мебель, на которую можно залезть. Всё собирается в
	// ЭТОМ ЖЕ проходе по view(7): второй обход обошёлся бы серверу дороже
	// всей остальной сенсорики вместе.
	var/obj/item/booze_src = null
	var/booze_dist = 999
	var/atom/water_src = null
	var/water_dist = 999
	var/obj/structure/climb_src = null
	var/climb_dist = 999
	var/water_best = 0

	for(var/turf/T as anything in seen)
		var/idx = flybrain_cell_index(T.x - origin.x, T.y - origin.y, facing)
		if(idx < 0)
			continue
		idx += 1        // списки в DM с единицы

		light[idx] = T.get_lumcount()
		solid[idx] = T.density ? 1 : 0

		var/m = 0
		for(var/mob/living/L in T)
			if(L == H)
				continue
			m = max(m, L.stat == DEAD ? 0.35 : 1)
			if(L.stat != DEAD)
				// крупное шевелящееся рядом — это угроза
				var/d = get_dist(H, L)
				if(d <= 3)
					danger_best = max(danger_best, 1 - d / 4)
				if(d < threat_dist)
					threat_dist = d
					threat = L
		mobs[idx] = m

		var/it = 0
		var/td = get_dist(H, T)
		var/near01 = td <= FLYBRAIN_SMELL_RADIUS ? (1 - td / (FLYBRAIN_SMELL_RADIUS + 1)) : 0
		for(var/obj/item/I in T)
			it = max(it, 0.6)
			if(!istype(I, /obj/item/weapon/reagent_containers))
				continue
			var/obj/item/weapon/reagent_containers/RC = I
			if(!near01)
				continue
			if(istype(I, /obj/item/weapon/reagent_containers/food))
				food_best = max(food_best, near01)
				if(td < food_dist)
					food_dist = td
					food_src = I
			// Брожение. Дрозофила летит на спирт — это не шутка, а один из
			// самых изученных её аппетитов. Ищем этанол в содержимом.
			if(flybrain_has_booze(RC))
				if(td < booze_dist)
					booze_dist = td
					booze_src = I
			if(flybrain_has_water(RC))
				water_best = max(water_best, near01)
				if(td < water_dist)
					water_dist = td
					water_src = I
		items[idx] = it

		// Вода на полу: лужи и мокрый пол — то, к чему муха тянется по ppk28.
		if(near01 && flybrain_wet_turf(T))
			water_best = max(water_best, near01)
			if(td < water_dist)
				water_dist = td
				water_src = T

		// Мебель: отрицательный геотаксис. Муха ползёт вверх.
		if(near01)
			for(var/obj/structure/S in T)
				if(!flybrain_climbable(S))
					continue
				if(td < climb_dist)
					climb_dist = td
					climb_src = S

		if(T.on_fire_check())
			danger_best = max(danger_best, 1)

	// --- телесное состояние ------------------------------------------
	var/health01 = H.maxHealth > 0 ? clamp(H.health / H.maxHealth, 0, 1) : 0
	// Боль — прирост урона за тик. Кислородное голодание сюда не берём:
	// оно всё время плавает от дыхания и давало бы фоновую "боль".
	// Удушье уходит отдельным полем и попадает в CO2-канал.
	var/pain = 0
	var/dmg_now = H.getBruteLoss() + H.getFireLoss() + H.getToxLoss()
	if(last_damage >= 0)
		pain = clamp((dmg_now - last_damage) / 15, 0, 1)
	last_damage = dmg_now

	var/suffocate = clamp(H.getOxyLoss() / 50, 0, 1)

	// Голод шлём сырым числом сытости: масштабировать удобнее на питоновской
	// стороне, там это правится без пересборки мира. Поле hunger оставлено
	// для совместимости, но считается по игровым порогам, а не делением на
	// NUTRITION_LEVEL_FULL: свежий человек рождается с NUTRITION_LEVEL_NORMAL,
	// и при делении на FULL он "голоден на 40%" сразу после спавна.
	var/nutrition_now = -1
	var/hunger = 0
	var/sat = H.get_satiation()
	if(!isnull(sat))
		nutrition_now = sat
		var/span = NUTRITION_LEVEL_WELL_FED - NUTRITION_LEVEL_STARVING
		hunger = clamp((NUTRITION_LEVEL_WELL_FED - sat) / span, 0, 1)

	// --- атмосфера ----------------------------------------------------
	var/temp = T20C
	var/pressure = ONE_ATMOSPHERE
	var/co2 = 0
	var/toxin = 0
	var/datum/gas_mixture/env = origin.return_air()
	if(env)
		temp = env.temperature
		pressure = env.return_pressure()
		var/total = max(env.get_total_moles(), 0.01)
		co2 = clamp(env.get_gas("carbon_dioxide") / total * 12, 0, 1)
		// в TauCeti плазма называется phoron
		toxin = clamp(env.get_gas("phoron") / total * 20, 0, 1)

	var/wind = clamp(abs(pressure - ONE_ATMOSPHERE) / ONE_ATMOSPHERE, 0, 1)

	// --- контакт и захваты --------------------------------------------
	var/contact = 0
	if(H.pulledby)
		contact = 1
	else if(bumped_recently)
		contact = 0.7
	// Запоминаем ДО сброса: направление удара нужно ниже, чтобы
	// муха чистила именно тронутое место.
	var/hit_dir = bumped_recently ? bump_dir : 0
	bumped_recently = FALSE
	bump_dir = 0

	// --- еда в руках и под ногами -------------------------------------
	var/food_touch = 0
	var/bitter = 0
	for(var/obj/item/I in list(H.l_hand, H.r_hand))
		if(istype(I, /obj/item/weapon/reagent_containers/food))
			food_touch = 1
		else if(I)
			bitter = max(bitter, 0.3)

	// Тарзальный контакт: у мухи вкусовые сенсиллы на лапках, и именно
	// "наступила на еду" запускает вытягивание хоботка.
	var/food_underfoot = 0
	for(var/obj/item/weapon/reagent_containers/food/F in origin)
		food_underfoot = 1
		break

	// Смещение угрозы и еды в системе координат мухи: fx вправо, fy вперёд.
	// Угол считает питон — в DM двухаргументного arctan лучше не трогать,
	// в кодовой базе он нигде не используется.
	var/threat_fx = 0
	var/threat_fy = 0
	var/threat_near = 0
	if(threat)
		var/list/tf = flybrain_body_frame(threat.x - origin.x, threat.y - origin.y, facing)
		threat_fx = tf[1]
		threat_fy = tf[2]
		threat_near = round(clamp(1 - threat_dist / 8, 0, 1), 0.01)

	var/food_fx = 0
	var/food_fy = 0
	if(food_src)
		var/turf/FT = get_turf(food_src)
		if(FT)
			var/list/ff = flybrain_body_frame(FT.x - origin.x, FT.y - origin.y, facing)
			food_fx = ff[1]
			food_fy = ff[2]

	// Пеленги на выпивку, воду и мебель — одной процедурой, чтобы не
	// размножать один и тот же поворот координат четыре раза.
	var/list/bz = flybrain_bearing_to(booze_src, origin, facing)
	var/list/wt = flybrain_bearing_to(water_src, origin, facing)
	var/list/cl = flybrain_bearing_to(climb_src, origin, facing)

	// Обратное расстояние считаем заранее: переносить строки внутри списка
	// обратным слэшем в DM — верный способ получить непонятную ошибку.
	var/booze_near = 0
	if(booze_dist <= FLYBRAIN_SMELL_RADIUS)
		booze_near = round(1 - booze_dist / (FLYBRAIN_SMELL_RADIUS + 1), 0.01)
	var/climb_near = 0
	if(climb_dist <= FLYBRAIN_SMELL_RADIUS)
		climb_near = round(1 - climb_dist / (FLYBRAIN_SMELL_RADIUS + 1), 0.01)

	// Спирт в организме и «на возвышении» — состояние тела, не окружения.
	var/drunk = flybrain_blood_alcohol(H)
	var/on_high = 0
	for(var/obj/structure/S in origin)
		if(flybrain_climbable(S))
			on_high = 1
			break

	// Куда её тронули. Щетинки у мухи сомато­топичны, и чистит она именно
	// тронутое место, поэтому направление важнее самого факта касания.
	var/touch_fx = 0
	var/touch_fy = 0
	if(H.pulledby)
		var/list/tt = flybrain_bearing_to(H.pulledby, origin, facing)
		touch_fx = tt[1]
		touch_fy = tt[2]
	else if(hit_dir)
		var/list/bd = flybrain_body_frame(
			(hit_dir & EAST) ? 1 : ((hit_dir & WEST) ? -1 : 0),
			(hit_dir & NORTH) ? 1 : ((hit_dir & SOUTH) ? -1 : 0), facing)
		touch_fx = bd[1]
		touch_fy = bd[2]

	var/list/p = list(
		"mob"       = id,
		"secret"    = SSflybrain.secret,
		"tick"      = ticks,
		"view_w"    = FLYBRAIN_VIEW_SIDE,
		"view_h"    = FLYBRAIN_VIEW_SIDE,
		"light"     = flybrain_pack(light),
		"solid"     = flybrain_pack(solid),
		"mobs"      = flybrain_pack(mobs),
		"items"     = flybrain_pack(items),
		"health"    = round(health01, 0.01),
		"pain"      = round(pain, 0.01),
		"hunger"    = round(hunger, 0.01),
		"nutrition" = round(nutrition_now, 1),
		"suffocate" = round(suffocate, 0.01),
		"temp"      = round(temp, 0.1),
		"pressure"  = round(pressure, 0.1),
		"co2"       = round(co2, 0.01),
		"toxin"     = round(max(toxin, danger_best * 0.5), 0.01),
		"food_near" = round(food_best, 0.01),
		"food_touch"= food_touch,
		"food_underfoot" = food_underfoot,
		"food_fx"   = food_fx,
		"food_fy"   = food_fy,
		"water_near" = round(water_best, 0.01),
		"water_fx"  = wt[1],
		"water_fy"  = wt[2],
		"booze_near" = booze_near,
		"booze_fx"  = bz[1],
		"booze_fy"  = bz[2],
		"drunk"     = round(drunk, 0.01),
		"climb_near" = climb_near,
		"climb_fx"  = cl[1],
		"climb_fy"  = cl[2],
		"on_high"   = on_high,
		"touch_fx"  = touch_fx,
		"touch_fy"  = touch_fy,
		"bitter"    = round(bitter, 0.01),
		"wind"      = round(wind, 0.01),
		"sound"     = round(sound_level, 0.01),
		"contact"   = round(contact, 0.01),
		"threat_fx"  = threat_fx,
		"threat_fy"  = threat_fy,
		"threat_near"= threat_near,
		"on_fire"   = H.on_fire ? 1 : 0,
		"stunned"   = ((H.stat != CONSCIOUS || H.weakened) ? 1 : 0),
	)
	sound_level = max(0, sound_level - 0.3)
	return p

/// Огонь на клетке — по-разному называется в разных ветках, поэтому мягко.
/turf/proc/on_fire_check()
	for(var/obj/fire/F in src)
		return TRUE
	return FALSE
