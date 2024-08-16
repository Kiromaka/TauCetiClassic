
var/inner_tele_radius = 50
var/outer_tele_radius = 250
var/include_space = 0
var/include_dense = 0

/obj/item/weapons/melee/cursed_blade
	can_embed = 0
	name = "Cursed blade"
	desc = "Something something, very scary"
	icon = 'icons/obj/heretic/blades.dmi'
	icon_state = "blade0"
	force = 30.0
	throwforce = 30.0
	throw_speed = 1
	throw_range = 5
	w_class = SIZE_SMALL
	attack_verb = list("attacked", "slashed", "stabbed", "sliced", "torn", "ripped", "diced", "cut")
	edge = 1
	hitsound = 'sound/weapons/bladeslice.ogg'



/obj/item/weapons/melee/cursed_blade/examine(mob/user)
	. = ..()


/obj/item/weapons/melee/cursed_blade/attack_self(list/targets, mob/user = usr)
	. = ..()

	to_chat(user,("You shatter [src]."))

	playsound(src,'sound/antag/eminence_hit.ogg',VOL_EFFECTS_MASTER)

	for(var/mob/living/target in targets)
		var/list/turfs = list()
		for(var/turf/T in range(target,outer_tele_radius))
			if(T in range(target,inner_tele_radius)) continue
			if(isenvironmentturf(T) && !include_space) continue
			if(T.density && !include_dense) continue
			if(T.x>world.maxx-outer_tele_radius || T.x<outer_tele_radius)	continue	//putting them at the edge is dumb
			if(T.y>world.maxy-outer_tele_radius || T.y<outer_tele_radius)	continue
			if(SEND_SIGNAL(T, COMSIG_ATOM_INTERCEPT_TELEPORT))
				continue
			turfs += T

		if(!turfs.len)
			var/list/turfs_to_pick_from = list()
			for(var/turf/T in orange(target,outer_tele_radius))
				if(!(T in orange(target,inner_tele_radius)))
					turfs_to_pick_from += T
			turfs += pick(/turf in turfs_to_pick_from)

		var/turf/picked = pick(turfs)

		if(!picked || !isturf(picked))
			return

		target.forceMove(picked)

	qdel(src)

