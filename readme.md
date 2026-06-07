# PV Viewer — Workshop Starter Notebook

This notebook fetches archived (X)GMD and GDet data from the archive appliance and plots a single PV at a time. The widget in this notebook is interactive, allowing the user to zoom, pan, reset axis limits, and save the plot.

## Your assignment
*Scenario:* We would like to develop this widget into a standalone **PyDM / PyQt
application** that can be invoked through the LCLSHOME Launchpad. Your solution should:
1. Display all 4 plots in a GUI window. *(Hint: use `matplotlib` subplots)*
2. Let the user select a time range to plot. 
3. Have a *Refresh* button that re-fetches and redraws the data.
4. Preserve the interactive features of the widget: home (reset axes), forward/back, pan, zoom, save figure.

> **Tip:** Start by asking the agent to read this notebook and explain what each piece does,
> then ask it to scaffold a PyDM `Display` (or a plain PyQt `QWidget`) around the same logic.

If you quickly accomplish the above and have time to spare, you may try a few of these optional/"stretch" goals:
1. When opened, have the viewer default to fetching and plotting the last 8 hours of data.
2. Allow the user to switch between viewing a single PV and all PVs on the canvas. 
3. Include hotkeys for the widget controls above. E.g. `z` to toggle zoom mode, `p` to toggle pan, `r` to reset, left and right arrow keys to go back or forward between views.
4. Add a nicer date/time picker, such as combo boxes or a selectable calendar overlay.
5. Experiment with pulling periods greater than 8 hours. Downsample long series to around 1000 data points to speed up plotting.

> **Tip:** As the agent adds new features, it may inadvertently break earlier functionality. In the prompt or in
> an `AGENTS.md`, ask the agent to write a test for each feature it adds using a module such as `pytest`.

> **Tip:** You are free to use your own software design instincts to guide the agent, e.g., "make a class that represents
> the canvas and preserves the current axis limits as class variables." This can make it easier for both you and the agent to 
> understand and remember key features of the code for future development.

## Environment setup
One goal of this assignment is to demonstrate the advantage of local agents over cloud execution or copy/pasting code when working in specialized environments. This notebook queries the PV archiver, which only accepts requests from certain hosts. Therefore, an agent running the repo in its own sandbox or on a cloud server would be unable to meaningfully test its own code in this case. You may need to *tell* the agent (with either prompts or AGENTS.md) to run commands on the machine itself, not its own sandbox, and you'll need to approve (or auto-approve) the terminal commands it wants to run.

To standardize the archiver behavior, you should run this notebook on `dev-srv09`, add the following to the bottom of your `.bashrc`, and then `source .bashrc`:

```
# Source the production environment into the current shell.
prodondev() {
  source /afs/slac/g/lcls/tools/script/ENVS64.bash
  source $EPICS_SETUP/envSet_prodOnDev.bash 
}

prodondev
```

(For why this is necessary, see: https://confluence.slac.stanford.edu/spaces/ARD/pages/695784800/Prod-on-dev+setup)