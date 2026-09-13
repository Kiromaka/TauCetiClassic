/*
 *  Транспорт до питоновского демона.
 *
 *  Два варианта, отличаются только скоростью:
 *
 *  rustg — call() в нативную библиотеку, запрос уходит в фоновый поток Rust,
 *          DM его не ждёт вообще. Отправили на тике N, забрали на тике N+1.
 *          Стоимость в DM: два call() по несколько микросекунд.
 *
 *  file  — пишем пакет в data/flybrain/<id>.in, зовём world.ext_python(),
 *          он POST-ит в тот же демон и кладёт ответ в <id>.out.
 *          Работает на голом репозитории без единого бинарника, но
 *          shell() порождает процесс и усыпляет вызывающую процедуру,
 *          поэтому это 50-200 мс на тик и потолок в 2-5 Гц.
 */

/datum/flybrain_transport
	var/name = "abstract"
	var/url = "http://127.0.0.1:5665"
	var/secret = ""

/datum/flybrain_transport/New(url, secret)
	src.url = url
	src.secret = secret

/// Поставить запрос. Возвращает хэндл (или TRUE), FALSE при ошибке.
/datum/flybrain_transport/proc/send(datum/fly_pilot/P, body)
	return FALSE

/// Забрать ответ. null — ещё не готов, "" — ошибка, иначе тело ответа.
/datum/flybrain_transport/proc/poll(datum/fly_pilot/P)
	return ""

/datum/flybrain_transport/proc/available()
	return TRUE


// ---------------------------------------------------------------------------
// rust_g: асинхронный HTTP
// ---------------------------------------------------------------------------
/datum/flybrain_transport/rustg
	name = FLYBRAIN_TRANSPORT_RUSTG

/datum/flybrain_transport/rustg/available()
	return flybrain_rustg_available()

/datum/flybrain_transport/rustg/send(datum/fly_pilot/P, body)
	var/headers = json_encode(list(
		"Content-Type" = "application/json",
		"X-Flybrain-Secret" = secret,
	))
	var/id
	try
		id = flybrain_http_async("post", "[url]/tick", body, headers)
	catch(var/exception/e)
		P.fail("rust_g: [e]")
		return FALSE
	if(!id)
		return FALSE
	P.handle = id
	return id

/datum/flybrain_transport/rustg/poll(datum/fly_pilot/P)
	if(!P.handle)
		return ""
	var/res
	try
		res = flybrain_http_check(P.handle)
	catch(var/exception/e)
		P.handle = null
		P.fail("rust_g check: [e]")
		return ""
	if(!res || res == "")
		return null                       // ещё в полёте
	P.handle = null
	var/list/parsed
	try
		parsed = json_decode(res)
	catch
		return ""
	if(!islist(parsed))
		return ""
	if(parsed["status_code"] != 200)
		P.fail("HTTP [parsed["status_code"]]: [copytext("[parsed["body"]]", 1, 160)]")
		return ""
	return parsed["body"]


// ---------------------------------------------------------------------------
// файлы + ext_python
// ---------------------------------------------------------------------------
/datum/flybrain_transport/file
	name = FLYBRAIN_TRANSPORT_FILE

/datum/flybrain_transport/file/available()
	return !!config.python_path

/datum/flybrain_transport/file/send(datum/fly_pilot/P, body)
	var/in_file = "[FLYBRAIN_IO_DIR][P.id].in"
	var/out_file = "[FLYBRAIN_IO_DIR][P.id].out"
	if(fexists(in_file))
		fdel(in_file)
	if(fexists(out_file))
		fdel(out_file)
	text2file(body, in_file)
	P.handle = out_file
	// ext_python усыпляет ЭТУ процедуру, но не мир: остальные SS продолжают
	// тикать. Поэтому запуск заворачиваем в spawn и не ждём его в fire().
	spawn(0)
		world.ext_python("flybrain_client.py",
			"--in [in_file] --out [out_file] --url [url] --secret [secret]")
	return TRUE

/datum/flybrain_transport/file/poll(datum/fly_pilot/P)
	if(!P.handle)
		return ""
	if(!fexists(P.handle))
		return null
	var/txt = file2text(P.handle)
	fdel(P.handle)
	P.handle = null
	return txt
