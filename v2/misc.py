import curses

import torch

def spherical2cartesian(directions):
	t, p = directions.T

	return torch.vstack([
		t.sin() * p.cos(),
		t.sin() * p.sin(),
		t.cos()
	])



def select_(stdscr, options):
	curses.curs_set(0)  # Hide the cursor
	stdscr.clear()

	current_selection = 0

	while True:
		h, w = stdscr.getmaxyx()

		pad = lambda x: " "*(w//2 - len(x)//2) + x + " "*(w//2)

		stdscr.addstr(0, 0, pad("PICK ONE"), curses.A_REVERSE)

		for idx, option in enumerate(options):
			if idx == current_selection:
				stdscr.addstr(idx+1, 0, option, curses.A_REVERSE)  # Highlight selected option
			else:
				stdscr.addstr(idx+1, 0, option)

		key = stdscr.getch()

		if key == curses.KEY_UP and current_selection > 0:
			current_selection -= 1
		elif key == curses.KEY_DOWN and current_selection < len(options) - 1:
			current_selection += 1
		elif key == curses.KEY_ENTER or key in [10, 13]:  # Enter key
			break

		stdscr.refresh()

	return options[current_selection]

def select(options = ["Option 1", "Option 2", "Option 3"]):
	return curses.wrapper(select_, options)

def optim_lr(optimizer):
	if optimizer.param_groups:
		return optimizer.param_groups[0]['lr']

	return None
