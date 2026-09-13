/*
 *  Минимальная привязка к rust_g: нужен только HTTP-модуль.
 *
 *  Если в кодовую базу когда-нибудь заедет полноценный rust_g.dm,
 *  этот файл можно просто удалить — определения совместимы по именам.
 */

#ifndef RUST_G

var/global/__flybrain_rust_g

/proc/__flybrain_detect_rust_g()
	if(world.system_type == UNIX)
		if(fexists("./librust_g.so"))
			return global.__flybrain_rust_g = "./librust_g.so"
		if(fexists("./rust_g"))
			return global.__flybrain_rust_g = "./rust_g"
		var/home = world.GetConfig("env", "HOME")
		if(home && fexists("[home]/.byond/bin/rust_g"))
			return global.__flybrain_rust_g = "rust_g"
		return global.__flybrain_rust_g = "librust_g.so"
	if(fexists("./rust_g.dll"))
		return global.__flybrain_rust_g = "./rust_g.dll"
	return global.__flybrain_rust_g = "rust_g"

#define RUST_G (global.__flybrain_rust_g || __flybrain_detect_rust_g())

#endif

/// Проверка, что библиотека реально грузится. Вызывать один раз при старте:
/// если dll/so нет, call_ext() кинет рантайм, который мы здесь и ловим.
/proc/flybrain_rustg_available()
	var/static/checked = null
	if(checked != null)
		return checked
	checked = FALSE
	try
		var/id = call_ext(RUST_G, "http_request_async")("get", "http://127.0.0.1:1/", "", "", "")
		if(id)
			checked = TRUE
	catch
		checked = FALSE
	return checked

#define flybrain_http_async(method, url, body, headers) \
	call_ext(RUST_G, "http_request_async")(method, url, body, headers, "")

#define flybrain_http_check(id) call_ext(RUST_G, "http_check_request")(id)
