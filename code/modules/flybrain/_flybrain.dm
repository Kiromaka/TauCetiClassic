/*
 *  flybrain — цифровой двойник мозга дрозофилы за рулём человека в SS13.
 *
 *  Модуль ничего не знает про нейронауку: он собирает сенсорный пакет,
 *  отдаёт его наружу и исполняет пришедшую моторную команду.
 *  Всё остальное живёт в питоновском демоне.
 *
 *  Подключение: см. INSTALL.md. Ядро кодовой базы трогать не обязательно —
 *  правка world.dm нужна только для необязательного обратного канала.
 */

// --- транспорты ------------------------------------------------------------
#define FLYBRAIN_TRANSPORT_RUSTG  "rustg"   // асинхронный HTTP через rust_g, быстрый
#define FLYBRAIN_TRANSPORT_FILE   "file"    // через файлы + world.ext_python, медленный

// --- сенсорика -------------------------------------------------------------
#define FLYBRAIN_VIEW_RADIUS 7               // 15x15 клеток, как обычный экран
#define FLYBRAIN_VIEW_SIDE   (FLYBRAIN_VIEW_RADIUS * 2 + 1)
#define FLYBRAIN_VIEW_CELLS  (FLYBRAIN_VIEW_SIDE * FLYBRAIN_VIEW_SIDE)

// радиус, в котором муха "чует" еду и опасность
#define FLYBRAIN_SMELL_RADIUS 7

// --- тайминги --------------------------------------------------------------
#define SS_INIT_FLYBRAIN 1
#define SS_WAIT_FLYBRAIN 2                   // 0.2 сек между сенсорными тиками
#define FLYBRAIN_TIMEOUT  50                 // 5 сек: считаем запрос потерянным

// --- служебное -------------------------------------------------------------
#define FLYBRAIN_CONFIG_FILE "config/flybrain.txt"
#define FLYBRAIN_IO_DIR      "data/flybrain/"

/// Упаковать список 0..1 в строку цифр '0'-'9'.
/// 225 чисел превращаются в 225 символов вместо ~1.5 КБ JSON —
/// это критично для файлового транспорта, где всё летит через шелл.
/proc/flybrain_pack(list/vals)
	var/list/out = list()
	for(var/v in vals)
		out += ascii2text(48 + clamp(round(v * 9), 0, 9))
	return jointext(out, "")

/// Смещение (dx, dy) в системе координат мухи: list(вправо, вперёд).
/// Один и тот же поворот нужен и для угрозы, и для еды, и для поля зрения.
/proc/flybrain_body_frame(dx, dy, facing)
	switch(facing)
		if(SOUTH)
			return list(-dx, -dy)
		if(EAST)
			return list(-dy, dx)
		if(WEST)
			return list(dy, -dx)
	return list(dx, dy)

/// Индекс клетки поля зрения в системе координат мухи.
/// Строка 0 — самое дальнее "вперёд", столбец 0 — крайний левый.
/// Возвращает -1, если клетка вне поля.
/proc/flybrain_cell_index(dx, dy, facing)
	var/list/f = flybrain_body_frame(dx, dy, facing)
	var/fx = f[1]   // вправо от мухи
	var/fy = f[2]   // вперёд
	var/col = fx + FLYBRAIN_VIEW_RADIUS
	var/row = FLYBRAIN_VIEW_RADIUS - fy
	if(col < 0 || col >= FLYBRAIN_VIEW_SIDE || row < 0 || row >= FLYBRAIN_VIEW_SIDE)
		return -1
	return row * FLYBRAIN_VIEW_SIDE + col

// --- распознавание того, к чему муха тянется --------------------------------
// Всё ниже написано защитно: имена реагентов и структур в ветках кодовой
// базы разъезжаются, а падать из-за отсутствующего типа модуль не должен.
// Проверяем и по имени реагента, и по типу контейнера — что-нибудь да совпадёт.

/// Есть ли в контейнере спирт.
/proc/flybrain_has_booze(obj/item/weapon/reagent_containers/RC)
	if(!RC)
		return FALSE
	if(!RC.reagents || !RC.reagents.reagent_list)
		return FALSE
	for(var/datum/reagent/R in RC.reagents.reagent_list)
		var/id = "[R.id]"
		if(findtext(id, "ethanol") || findtext(id, "beer") || findtext(id, "wine"))
			return TRUE
		if(findtext(id, "vodka") || findtext(id, "whiskey") || findtext(id, "rum"))
			return TRUE
		if(findtext(id, "gin") || findtext(id, "tequila") || findtext(id, "cognac"))
			return TRUE
		if(findtext(id, "ale") || findtext(id, "mead") || findtext(id, "absinthe"))
			return TRUE
	return FALSE

/// Есть ли в контейнере вода.
/proc/flybrain_has_water(obj/item/weapon/reagent_containers/RC)
	if(!RC || !RC.reagents || !RC.reagents.reagent_list)
		return FALSE
	for(var/datum/reagent/R in RC.reagents.reagent_list)
		var/id = "[R.id]"
		if(findtext(id, "water") || findtext(id, "juice") || findtext(id, "tea"))
			return TRUE
	return FALSE

/// Мокрый пол или лужа. Проверяем мягко: имена переменных разные в ветках.
/proc/flybrain_wet_turf(turf/T)
	if(!T)
		return FALSE
	if(!isnull(T.vars["wet"]) && T.vars["wet"])
		return TRUE
	for(var/obj/effect/E in T)
		if(findtext("[E.type]", "water") || findtext("[E.type]", "puddle"))
			return TRUE
	return FALSE

/// Годится ли структура, чтобы залезть. Отрицательный геотаксис у мухи —
/// упорное ползание вверх, на этом построен стандартный тест на локомоцию.
/proc/flybrain_climbable(obj/structure/S)
	if(!S)
		return FALSE
	if(!isnull(S.vars["climbable"]))
		return S.vars["climbable"] ? TRUE : FALSE
	var/t = "[S.type]"
	if(findtext(t, "table") || findtext(t, "rack") || findtext(t, "bed"))
		return TRUE
	if(findtext(t, "closet") || findtext(t, "crate"))
		return TRUE
	return FALSE

/// Сколько спирта в организме, 0..1. Порог опьянения в SS13 около 50 единиц.
/proc/flybrain_blood_alcohol(mob/living/carbon/human/H)
	if(!H || !H.reagents || !H.reagents.reagent_list)
		return 0
	var/total = 0
	for(var/datum/reagent/R in H.reagents.reagent_list)
		var/id = "[R.id]"
		if(findtext(id, "ethanol") || findtext(id, "beer") || findtext(id, "wine"))
			total += R.volume
		else if(findtext(id, "vodka") || findtext(id, "whiskey") || findtext(id, "rum"))
			total += R.volume
	return clamp(total / 40, 0, 1)

/// Пеленг на объект в системе координат мухи: list(вправо, вперёд).
/// Возвращает list(0, 0), если объекта нет.
/proc/flybrain_bearing_to(atom/A, turf/origin, facing)
	if(!A || !origin)
		return list(0, 0)
	var/turf/T = get_turf(A)
	if(!T)
		return list(0, 0)
	return flybrain_body_frame(T.x - origin.x, T.y - origin.y, facing)
