# -*- coding: utf-8 -*-
"""
This code is used to automate PEMWE data analysis when working with a BioLogic potentiostat.
It supplements Hoffmeister et al.

getting help: https://setdown.rssing.com/chan-1811197/article65.html#c1811197a65

@author: Birk Fritsch and Selina Finger

@Dependencies:
This code is written with:
Python 3.10.6
NumPy 1.23.4
Matplotlib 3.6.2
SciPy 1.9.3
pandas 1.5.1
Impedance.py 1.7.1
Moreover, pandas depends on xlsxwriter or openpyxl for excel file writing. During development, xlsxwriter 3.1.9 was used.
"""

# imports
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from impedance.models.circuits import CustomCircuit
from impedance.models.circuits import elements


# add extra element to impedance.py
@elements.element(num_params=3, units=["Ohm", "F sec^(gamma - 1)", "Ohm"], overwrite=True)
def TLMA(p, f):

    """
    TLMQ for H2O2 cell with charge transfer --> Makharia, Mathias and Baker, https://iopscience.iop.org/article/10.1149/1.1888367
    """

    omega = 2 * np.pi * np.array(f)
    Rct, C, Rion = p[0], p[1], p[2]
    Zk = Rct / (1 + C * (1j * omega) * Rct)
    Z = np.sqrt(Rion * Zk) / np.tanh(np.sqrt(Rion / Zk))

    return Z


elements.TLMA = TLMA


# function definition
def calculate_r2(y, f, n, p):
    """
    Corrected (adjusted) R squared, after https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2

    Parameters
    ----------
    y : int or float
        values in data set.
    f : int or float
        fitted or predicted values for y.
    n : int or float
        sample size of data set.
    p : int or float
        number of explanatory variables in the model.

    Returns
    -------
    float
        Adjusted R squared so that R squared is not automatically
        increasing when extra explanatory variables are added to the model.

    """

    sstot = np.sum((y - y.mean()) ** 2)
    ssres = np.sum((y - f) ** 2)

    r_squared = 1 - ssres / sstot

    return 1 - (1 - r_squared) * (n - 1) / (n - p)


def detect_header_lines(filepath, sep="\t"):
    """
    Detects number of header lines in mpt file (that needs to be skipped later).
    This function assumes that the header lines contain less column separators
    than the actual data.

    Parameters
    ----------
    filepath : str or path-like
        The directory of the to-be-read mpt file including the filename itself

    optional:
        sep: str
            string that allows for separating columns. The default is \t.
    Returns
    -------
    sr : int
        The number of header rows to be skipped

    """
    # "with" ensures, that file is only opened while in this indent block
    with open(filepath) as file:
        # only to extract how many header lines are to be skipped
        contents = file.readlines()
        # skiprows, extracts the number from the second line (-1 to adjust to starting counting at 0)
        line_length = {}

        for n, line in enumerate(contents):
            line_list = line.split(sep)
            number_of_columns = len(line_list)
            if number_of_columns not in line_length:
                line_length[number_of_columns] = n
        sr = line_length[max(line_length)]

    return sr


def load_mpt_file(filepath):
    """
    loads an mpt file and stores it as a DataFrame

    Parameters
    ----------
    filepath : str or path-like
        The directory of the to-be-read mpt file including the filename itself

    Returns
    -------
    measurement_data : pd.DataFrame
        a 'raw' DataFrame containing the read information without preprocessing.

    """
    # load data using pandas
    sr = detect_header_lines(filepath)

    # create a pd.DataFrame object to store the information of interest
    measurement_data = pd.read_csv(filepath, sep="\t", decimal=",", skiprows=sr, encoding="ANSI")

    return measurement_data


def convert_raw_to_si(df, area):
    """
    shift starting point in 'time/s', and calculate 'time/h', i/A, 'i/A*cm^-2', 'Re(Z)/Ohm*cm^2' as well as '-Im(Z)/Ohm*cm^2'

    Parameters
    ----------
    df : pd.DataFrame
        as retreived from load_mpt_file
    area : int or float,
            The active area onto which the experimental data needs to be normalized to.
            Its unit is cm².

    Returns
    -------
    df : pd.DataFrame
        as input df, but with modified 'time/s', and added 'time/h', 'i/A', 'Re(Z)/mOhm*cm^2',
        '-Im(Z)/mOhm*cm^2', as well as 'i/A*cm^-2' columns.

    """

    df = df.copy()

    # reset time to start at 0
    df["time/s"] = df["time/s"] - df["time/s"][0]
    # add column to have time in hours
    df["time/h"] = df["time/s"] / 3600
    # calculate absolute current density i in A/cm²
    try:
        df["i/A"] = df["<I>/mA"] / 1000
    except KeyError:
        # if '<I>/mA' is not available, try 'I/mA'
        df["i/A"] = df["I/mA"] / 1000
        df["i_control/A*cm^-2"] = df["control/mA"] / (1000 * area)

    df["i/A*cm^-2"] = df["i/A"] / area
    # normalize resistance to area and to milli Ohm
    try:
        df["Re(Z)/mOhm*cm^2"] = df["Re(Z)/Ohm"] * area * 1000
        df["-Im(Z)/mOhm*cm^2"] = df["-Im(Z)/Ohm"] * area * 1000
    except KeyError as e:
        print("I did not find", e)

    return df


def split_ACDC(df):
    """
    splits a pandas dataframe on zero values in column 'freq/Hz' using conditional
    selection

    Parameters
    ----------
    df : pd.DataFrame
        as retreived from load_mpt_file

    Returns
    -------
    ac_df : pd.DataFrame
        all data with a frequency != 0
    dc_df : pd.DataFrame
        all data with frequency == 0
    """
    # precondition: df has a column called "freq/Hz". We split based on it
    cond = df["freq/Hz"] == 0
    # get everything with alternating current. ~ inverts cond
    ac_df = df[~cond]
    # get everything else (should be direct current)
    dc_df = df[cond]

    return ac_df, dc_df


def get_cycle_numbers(df, numbers_per_repetition):
    """
    Extracts the unique values from the df columns 'cycle number' and raises
    an error if its maximum value is not divisible by numbers_per_repetition

    Parameters
    ----------
    df : pd.DataFrame
        Must provide a 'cycle number' column
    numbers_per_repetition : int

    Raises
    ------
    ValueError
        gives error when modulus of cycle number and numbers_per_repetition
        is not 0 --> error in measurement can be directly detected.


    Returns
    -------
    cycle_numbers : array-like
        returns a np.array(?) containing the unique values in df['cycle number'].
    """

    # get unique values
    cycle_numbers = df["cycle number"].unique()
    # get maximum value
    max_cycle_number = cycle_numbers.shape[0]
    # test consistency of cycle number and numbers_per_repetition:

    if (max_cycle_number % numbers_per_repetition) != 0:
        # raise an exception
        raise ValueError(
            f"Numbers per repetition ({numbers_per_repetition}) does not match numbers of cycles ({max_cycle_number})."
        )

    return cycle_numbers


def convert_cycle_number_to_repetition(cycle_number, numbers_per_repetition):
    """
    Converts cycle number to repetitions for each entry via cycle_number // numbers_per_repetition + 1

    Parameters
    ----------
    cycle_number : integer or float
        Number of the measurment cycle. From raw data
        (=current density steps multiplied by repetitions)
    numbers_per_repetition : integer or float
        number of current density steps

    Returns
    -------
    integer
    """

    # as cycle number starts with 1, reduce by 1:
    return cycle_number // numbers_per_repetition + 1


def provide_data(file_name, area):
    """
    Load mpt file and convert key parameters to SI units

    Parameters
    ----------
    file_name : str
        file name to be passed to load_mpt_file.
    area : int or float,
            The active area onto which the experimental data needs to be normalized to.
            Its unit is cm².

    Returns
    -------
    converted_data : pd.DataFrame
        DataFrame object containing experimental data as retreived from convert_raw_to_si

    """
    raw_measurement_data = load_mpt_file(file_name)
    # shift starting point, add time in h, calculate current density, and normalize restistance
    converted_data = convert_raw_to_si(raw_measurement_data, area)

    return converted_data


def single_equivalent_circuit_fit(ac_data, equivalent_circuit, initial_circuit_guess):
    """
    Performs a least-squares optimization of an equivalent circuit model to EIS data.
    This relies on impedance.py. Please refer to this package for further details.

    It needs a defined equivalent circuit fit and initial guesses to return
    the fitted values for all circuit element including error to be stored
    in dictionary.

    Parameters
    ----------
    ac_data : pd.DataFrame
        all data with a frequency != 0
    equivalent_circuit : str
        specifies circuit elements for curve fitting. Implemented elements in
        impedance.py can be used in parallel or series
        See impedance.py for further syntax details.
    initial_circuit_guess : tuple
        initial values for each circuit element to be fed into fit

    Returns
    -------
    fitresult : dict
        contains objects from fit thta we are interested in:
    circuit_fit : np.aaray
        fits impedance model (with equivalent circuit) to data

    """
    # instantiate equivalent circuit object
    circuit = CustomCircuit(equivalent_circuit, initial_guess=initial_circuit_guess)
    # fit impedance model to data
    circuit.fit(np.array(ac_data["freq/Hz"]), np.array(ac_data["Z/Ohm"]))
    # calculate predicted values for nyquist plot
    circuit_fit = circuit.predict(np.array(ac_data["freq/Hz"]))
    """
    In order to access fit parameters from circuit, use
    circuit.get_param_names() to get the decription,
    circuit.parameters_ to access the fit outcomes, and
    circuit.conf_ to get the respective errors (std)
    """
    # get output out of fitted circuit. We are interested in hfr_object. --> fitresult[hfr_object]
    element, unit = circuit.get_param_names()
    fitresult = {elem: (val, err) for elem, val, err in zip(element, circuit.parameters_, circuit.conf_)}

    # gauge goodness of fit by R²_adj, similar to Origin.
    r2adj_real = calculate_r2(
        np.real(ac_data["Z/Ohm"]), np.real(circuit_fit), circuit_fit.shape[0], len(initial_circuit_guess)
    )

    r2adj_imag = calculate_r2(
        np.imag(ac_data["Z/Ohm"]), np.imag(circuit_fit), circuit_fit.shape[0], len(initial_circuit_guess)
    )
    # store r2adj in output dictionary
    fitresult["R2adj"] = r2adj_real, r2adj_imag

    return fitresult, circuit_fit


def store_lists_in_excel(storage_lists, column_names, savename, numbers_per_repetition, max_cycle_number):
    """
    Converts a nested list into a pd.DataFrame and stores it as an excel sheet
    with 'savename' as file name.

    Parameters
    ----------
    storage_lists : list
        Nested 2D lists.
    column_names : list
        List of names for columns with respective unit. Must match the the
        number of sublists in storage_lists. Should include '"'Cycle Number'
    savename : str or path-like
        file name for excel file
    numbers_per_repetition : int
        numbers per repetition of measurement
    max_cycle_number : int
        amount of measurement per cycle

    Returns
    -------
    summary_df : pd.DataFrame
        converted pd.DataFrame

    """
    # create dataframe
    summary_df = pd.DataFrame(storage_lists)
    # transpose dataframe
    summary_df = summary_df.T
    # name columns
    summary_df.columns = column_names
    # convert a cycle number to a run
    summary_df["Run"] = [convert_cycle_number_to_repetition(i, numbers_per_repetition) for i in range(max_cycle_number)]

    # set Cycle Number as index
    try:
        summary_df.set_index("Cycle Number", inplace=True)
    except ValueError:
        print('WARNING: "Cylcle Number cannot be set as index. Continuing.')

    # store as excel sheet
    summary_df.to_excel(f"{savename}.xlsx")

    return summary_df


def tafel_equation(i, a, b):
    """
    Fit equation to extract the Tafel slope from the low current density region

    Parameters
    ----------
    i : float
        current density from polarization curve.
    a : float
        y-axis intercept.
    b : float
        Tafel slope.

    Returns
    -------
    float
        potential.

    """
    return a + b * np.log10(i)


def polarization_curve(file_name, numbers_per_repetition, steady_state_length, area, show=True, savename=None):
    """
    Extracts data required for a polarization curve, stores them in an excel file
    and returns the related data as pd.DataFrame.

    Parameters
    ----------
    file_name : str or path-like
        Directory of the raw data in mpt format from biologic potentiostat
    numbers_per_repetition : int, optional
        Number of current density steps in measurement. The default is 18.
    steady_state_length : Int, optional
        Number of last data points in measurement that are averaged. The default is 30.
    area : int, float
        Active cell area in cm².
    show : boolean, optional
        Plots the polarization curve and stores it as png, svg and pdf. The default is True.
    savname : str or None, optional
        Name for image saving. None causes an automated name-building based on file_name. The default is None

    Returns
    -------
    summary_df : pd.DataFrame
        data frame with the columns
        ['Cycle Number', 'Current Denstiy / A cm-2', 'Current Density error / A cm-2',
        'Potential / V', 'Potential error / V'].
    """
    # load data and perform preprocessing
    if type(file_name) == str:
        measurement_data = provide_data(file_name, area)
    else:
        measurement_data = file_name
    # crop frequency == 0
    __, dc_data = split_ACDC(measurement_data)
    # split repetitions based on cycle numbers
    cycle_numbers = get_cycle_numbers(dc_data, numbers_per_repetition)
    # get amount of measurements per cycle
    max_cycle_number = cycle_numbers.shape[0]

    ##average last 30 data points for each cycle number
    # create output steady state (ss) containers
    ss_potential = []
    ss_potential_error = []
    ss_current_density = []
    ss_current_density_error = []
    for cycle_num in cycle_numbers:
        # get subset
        df = dc_data[dc_data["cycle number"] == cycle_num]
        # get steady state
        ss_values = df.iloc[-steady_state_length:]
        # get steady state potential
        ss_potential.append(ss_values["<Ewe>/V"].mean())
        # get sample standard deviation of steady state potential
        ss_potential_error.append(ss_values["<Ewe>/V"].std(ddof=1))
        # get steady state current density
        ss_current_density.append(ss_values["i/A*cm^-2"].mean())
        # get sample standard deviation of steady state current density
        ss_current_density_error.append(ss_values["i/A*cm^-2"].std(ddof=1))

    ###plot data
    if show:

        f, ax = plt.subplots(figsize=(5, 5), dpi=150)

        for run in range(max_cycle_number // numbers_per_repetition):
            # select limits
            lower_index = run * numbers_per_repetition
            run += 1
            upper_index = run * numbers_per_repetition

            ax.errorbar(
                x=ss_current_density[lower_index:upper_index],
                y=ss_potential[lower_index:upper_index],
                xerr=ss_current_density_error[lower_index:upper_index],
                yerr=ss_potential_error[lower_index:upper_index],
                fmt=".-",
                label=run,
            )
        ax.set(xlabel=r"$i_\mathrm{steady\,state}$ / A$\cdot$cm$^{-2}$", ylabel=r"$E_\mathrm{steady\,state}$ / V")

        ax.tick_params(direction="in", which="both")
        ax.legend(loc=0, title="Run")
        if not savename:
            # remove .mpt
            savename = file_name.split(".")[0]
        # append info
        for end in ["png", "svg", "pdf"]:
            f.savefig(f"{savename}_polcurve.{end}", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close("all")

    ###return data
    # store a summary as DataFrame and excel sheet
    summary_df = store_lists_in_excel(
        [cycle_numbers, ss_current_density, ss_current_density_error, ss_potential, ss_potential_error],
        [
            "Cycle Number",
            "Current Density / A cm-2",
            "Current Density error / A cm-2",
            "Potential / V",
            "Potential error / V",
        ],
        f"{savename}_polcurve",
        numbers_per_repetition,
        max_cycle_number,
    )

    return summary_df


def nyquist_plot(
    file_name,
    area,
    show=True,
    numbers_per_repetition=18,
    equivalent_circuit="L0-R0-TLMA0",
    initial_circuit_guess=(0.00001, 0.01, 0.01, 0.1, 0.01),
    frequency_interval=(0, np.inf),
):
    """
    Extracts data required for the nyquist plots, stores them in an excel file,
    and performs alls nyquist fits (See single_equivalent_circuit_fit for details).

    It returns the related data as pd.DataFrame.

    Parameters
    ----------
    file_name : str or path-like
        Directory of the raw data in mpt format from biologic potentiostat
    area :
    show : boolean, optional
        Plots the polarization curve and stores it as png, svg and pdf. The default is True.
    numbers_per_repetition : int, optional
        Number of current density steps in measurement. The default is 18.
    equivalent_circuit : str, optional
        Equivalent circuit model with individual elements. The default is 'L0-R0-p(R1,C1)-p(R2-Wo1,C2)'.
        To be passed to single_equivalent_circuit_fit, which makes use of impedance.py
    initial_circuit_guess : list, optional
        initial guesses for circuit elements. The amount of values must correspond
        to the chosen equivalent circuit model.
        The default is (0.01, 80.00e-9, 1, 0.01, 0.003, 0.01, 0.01, 0.01).
    frequency_interval : tuple or list,
        Tuple giving lower and upper frequency limit for fitting. Default is [0, np.inf].

    Raises
    ------
    ValueError
        gives error when modulus of cycle number and numbers_per_repetition
        is not 0 --> error in measurement can be directly detected.

    Returns
    -------
    summary_df : pd.DataFrame
        dataframe with value and error of all circuit elements,
        adjusted R squared, cycle number and run

    """

    # load data and perform preprocessing
    measurement_data = provide_data(file_name, area=area)
    summary_df = _nyquist_plot_core(
        measurement_data,
        file_name,
        show,
        numbers_per_repetition,
        equivalent_circuit,
        initial_circuit_guess,
        frequency_interval,
    )

    return summary_df


def _nyquist_plot_core(
    measurement_data,
    file_name,
    show=True,
    numbers_per_repetition=18,
    equivalent_circuit="L0-R0-TLMA0",
    initial_circuit_guess=(0.00001, 0.01, 0.01, 0.1, 0.01),
    frequency_interval=(0, np.inf),
):
    """
    see nyquist_plot
    """
    # crop frequency != 0
    ac_data, __ = split_ACDC(measurement_data)
    # split repetitions based on cycle numbers
    cycle_numbers = ac_data["cycle number"].unique()
    max_cycle_number = cycle_numbers.shape[0]
    # test consistency of cycle number and numbers_per_repetition:

    if (max_cycle_number % numbers_per_repetition) != 0:
        # raise an exception
        raise ValueError(
            f"Numbers per repetition ({numbers_per_repetition}) does not match numbers of cycles ({max_cycle_number})."
        )

    # summarize real and imaginary impedance in complex numbers
    ac_data["Z/Ohm"] = [complex(re, im) for re, im in zip(ac_data["Re(Z)/Ohm"], -ac_data["-Im(Z)/Ohm"])]

    ##prepare loop
    # create output containers
    output_dict = {}
    # prepare storage file name
    savename = file_name.split(".")[0] + "_Nyq_Fit"

    # dummy variable for first iteration
    new_fit_guess = initial_circuit_guess
    for cycle_num in cycle_numbers:
        # get subset
        df = ac_data[ac_data["cycle number"] == cycle_num]
        # limit frequency range for analysis
        lower_limit, upper_limit = frequency_interval
        df = df[(df["freq/Hz"] >= lower_limit) & (df["freq/Hz"] <= upper_limit)]
        ##fit impedance model to data. Impedance.py does only accept nd.array dtype
        # store results of loop in dictionary
        fit_ranking_dct = {}
        # loop flag definition
        successful_fit = False
        for i in range(2):

            # get eviqualent circuit information
            fitresult, circuit_fit = single_equivalent_circuit_fit(df, equivalent_circuit, new_fit_guess)
            # average real and imaginary part of R²
            mean_r2 = np.sqrt(np.mean(np.square(fitresult["R2adj"])))
            fit_ranking_dct[mean_r2] = fitresult, circuit_fit, mean_r2

            # check if fit was successful
            if mean_r2 >= 0.99:
                successful_fit = True
                break
            else:
                # if not: reset initial guesses
                new_fit_guess = initial_circuit_guess

        # retrieve best fit if all trials were bad:
        if not successful_fit:
            # get maximum key
            max_r2 = max(fit_ranking_dct)
            # unpack best values
            fitresult, circuit_fit, mean_r2 = fit_ranking_dct[max_r2]

        """
        Now, we need to retrieve the fit parameters as list from a dictionary.
        We rely on the fact that newer Python versions maintain the order within
        a dictionary. Also, we toss the R² which is the last entry.
        """
        new_fit_guess = [np.mean((val, err)) for val, err in fitresult.values()][:-1]

        # fill dictionary. During first run, lists must be created, too!
        for element in fitresult:
            if element not in output_dict:
                # create lists for each circuit element and include first value
                if element == "R2adj":
                    # take mean value of real and imaginary part
                    output_dict[element] = [mean_r2]
                else:
                    # create value list with first entry
                    output_dict[element] = [fitresult[element][0]]
                    # create error list with first entry
                    output_dict[f"{element} error"] = [fitresult[element][1]]
            else:
                if element == "R2adj":
                    # append mean value of real and imaginary part
                    output_dict[element].append(mean_r2)
                else:
                    # append value
                    output_dict[element].append(fitresult[element][0])
                    # append error
                    output_dict[f"{element} error"].append(fitresult[element][1])

        # plot the individual fits
        if show:
            f, ax = plt.subplots(figsize=(5, 5), dpi=150)
            # plot experimental data
            c = ax.scatter(
                x="Re(Z)/Ohm",
                y="-Im(Z)/Ohm",
                c="freq/Hz",
                marker=".",
                # ls='',
                data=df,
                label=f"Run {int(cycle_num)}",
                cmap="viridis",
                norm=LogNorm(),
            )

            f.colorbar(c, ax=ax, label="Frequency / Hz")
            # plot fitresult
            ax.plot(
                np.real(circuit_fit),
                -np.imag(circuit_fit),
                color="red",
                label=f"$R^2_\\mathrm{{adj}}={round(mean_r2, 5)}$",
            )
            ax.set(xlabel=r"$\Re(Z)$ / $\Omega$", ylabel=r"$-\Im(Z)$ / $\Omega$")
            ax.axhline(0, color="gray", ls=":", zorder=0)
            # enforce equal scaling for x and y
            ax.axis("equal")
            ax.tick_params(direction="in", which="both")
            ax.legend(loc=0)
            # save
            for end in ["png", "svg", "pdf"]:
                f.savefig(f"{savename}_Cyc_{int(cycle_num)}.{end}", dpi=300, bbox_inches="tight")

            plt.show()
            plt.close("all")
    # summarize in dataframe and store as excel sheet
    summary_df = store_lists_in_excel(
        [
            cycle_numbers,
            *output_dict.values(),
        ],
        [
            "Cycle Number",
            *output_dict.keys(),
        ],
        savename,
        numbers_per_repetition,
        max_cycle_number,
    )
    return summary_df


def hfr_evolution(polcurve_df, nyquist_df, area, hfr_element="R0", savename="test"):
    """
    Transforms HFR into mOhm cm2, calulates HFR-free potential,
    merges polcurve_df and nyquist_df, stores them in an excel file
    and returns the related data as pd.DataFrame.

    Parameters
    ----------
    polcurve_df : pd.DataFrame
        dataframe with relevant data to plot polarization curve.
    nyquist_df : pd.DataFrame
        dataframe with relevant data to fit and plot nyquist plot.
    area : int or float,
            active area of cell used in measurement in cm².
    hfr_element : str, optional
        equivalent circuit element that is of interest. The default is 'R0'.
    savename : str, optional
        name under which excel file and plots are saved. The default is 'test'.

    Returns
    -------
    df : pd.DataFrame
        dataframe containing merged values from polcurve_df, nyquist_df
        as well as value and error of HFR, HFR potential and HFR-free potential.

    """

    # calculate new values
    nyquist_df["HFR / mOhm cm2"] = nyquist_df[hfr_element] * area * 1000
    nyquist_df["HFR error / mOhm cm2"] = np.abs(nyquist_df[f"{hfr_element} error"] * area * 1000)

    # merge dataframes for simplicity
    df = pd.merge(polcurve_df.drop("Run", axis=1), nyquist_df, left_index=True, right_index=True)

    # calulate HFR-free potential
    df["HFR Potential / V"] = df["HFR / mOhm cm2"] * 1e-3 * df["Current Density / A cm-2"]
    # calculate HFR-free potential error after Gaussian error propagation
    df["HFR Potential error / V"] = np.abs(df["HFR error / mOhm cm2"] * 1e-3 * df["Current Density / A cm-2"]) + np.abs(
        df["Current Density error / A cm-2"] * df["HFR / mOhm cm2"] * 1e-3
    )

    df["HFR-free Potential / V"] = df["Potential / V"] - df["HFR Potential / V"]
    # calculate HFR-free potential error after Gaussian error propagation
    df["HFR-free Potential error / V"] = df["Potential error / V"] + df["HFR Potential error / V"]

    savename += "_withHFR"

    df.to_excel(savename + ".xlsx")

    return df


def store_ac_data_as_excel(df, runs, save_name="test"):
    """
    Stores alternating current (ac) measurement data as Excel file (xlsx).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a "cycle number" column that can be passed to convert_cycle_number_to_repetition.
    runs : int
        Amount of ac repetitions. Is passed to convert_cycle_number_to_repetition.
    save_name : str, optional
        Name for data saving. Do not attach the "xlsx" ending here, it will be added automatically. The default is 'test'.

    Returns
    -------
    None.

    """
    ac_df, __ = split_ACDC(df)
    # tbd: create one spreadsheet per run
    ac_df["Run"] = ac_df["cycle number"].apply(lambda x: convert_cycle_number_to_repetition(runs, x))
    for run in ac_df["Run"].unique():
        run_df = ac_df[ac_df["Run"] == run]
        run_df.to_excel(f"{save_name}_{run}.xlsx")


def plot_hfr_evolution(hfr_df, savename="test"):
    """
    plots HFR values and error over current density steps

    Parameters
    ----------
    hfr_df : pd.DataFrame
        dataframe with values from polcurve_df, nyquist_df and df (from hfr_evolution)
    savename : str, optional
        name under which figure is saved as png, svg and pdf. The default is 'test'.

    Returns
    -------
    None.

    """

    # plot hfr vs current density
    f_hfr_free, (ax_pol, ax) = plt.subplots(
        2, 1, figsize=(5, 6.5), dpi=150, layout="constrained", sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )

    # plot every run separartely
    for color, run in zip(["C0", "C1", "C2"], hfr_df["Run"].unique()):

        df_run = hfr_df[hfr_df["Run"] == run]

        # plot E_steady state vs i_steady state on other figure
        ax_pol.errorbar(
            x="Current Density / A cm-2",
            xerr="Current Density error / A cm-2",
            y="Potential / V",
            yerr="Potential error / V",
            label=f"{run}",
            data=df_run,
            ls="-",
            marker=".",
        )

        ax_pol.errorbar(
            x="Current Density / A cm-2",
            xerr="Current Density error / A cm-2",
            y="HFR-free Potential / V",
            yerr="HFR-free Potential error / V",
            label=f"{run} HFR-free",
            data=df_run,
            c=color,
            ls="-",
            marker="x",
        )

        # plot hfr vs current density
        ax.errorbar(
            x="Current Density / A cm-2",
            xerr="Current Density error / A cm-2",
            y="HFR / mOhm cm2",
            yerr="HFR error / mOhm cm2",
            label=f"{run}",
            data=df_run,
            c=color,
            ls="-",
            marker=".",
        )

    # layout hfr plot
    ax.set(xlabel=r"$i_\mathrm{steady\,state}$ / A cm$^{-2}$", ylabel=r"HFR / m$\Omega$ cm$^2$")

    # layout hfr-free polcurve plot
    ax_pol.legend(loc=0, fontsize="x-small", ncol=3, title="Run", title_fontsize="small")
    ax_pol.set(ylabel=r"$E_\mathrm{steady\,state}$ / V")

    ylims = ax_pol.get_ylim()
    ax_pol.set_ylim((ylims[0], ylims[1] * 1.05))

    for a in [ax, ax_pol]:
        a.tick_params(direction="in", which="both")

    for end in ["png", "svg", "pdf"]:
        f_hfr_free.savefig(f"{savename}_+hfr-free.{end}", dpi=300)

    plt.show()


def tafel_plot(hfr_df, run=1, fit_limit=0.1, savename="test"):
    """
    Fit Tafel equation to HFR-free polarization curve, plot polarization curve with fit,
    extract Tafel slope and show it in legend.

    Parameters
    ----------
    hfr_df :  pd.DataFrame
        dataframe with values from polcurve_df, nyquist_df and df (from hfr_evolution)
    run : int, optional
        number of runs (i.e. repetitions of same sequence) in measurement.
        The default is 1.
    fit_limit : float, optional
        upper current density limit for fitting. The default is 0.1.
    savename : str, optional
       name under which figure is saved as png, svg and pdf. The default is 'test'.

    Returns
    -------
    list
        list with best fit (least-squares fit) parameters popt and
        covariance matrix pcov, which indicates the uncertainties
        and correlations between parameters

    """

    # make figure canvas
    f, ax = plt.subplots(
        figsize=(5, 5),
        dpi=150,
        layout="constrained",
    )

    df_run = hfr_df[hfr_df["Run"] == run]

    ax.errorbar(
        x="Current Density / A cm-2",
        xerr="Current Density error / A cm-2",
        y="HFR-free Potential / V",
        yerr="HFR-free Potential error / V",
        label=f"Run {run} HFR-free",
        data=df_run,
        ls="-",
        marker=".",
        zorder=0,
    )

    ax.set(xlabel=r"$i_\mathrm{steady\,state}$ / A cm$^{-2}$", ylabel=r"$E_\mathrm{steady\,state}$ / V", xscale="log")

    # do fitting
    fit_df = df_run[df_run["Current Density / A cm-2"] <= fit_limit]
    xdat = fit_df["Current Density / A cm-2"]
    ydat = fit_df["HFR-free Potential / V"]
    y_error = fit_df["HFR-free Potential error / V"]
    popt, pcov = curve_fit(tafel_equation, xdat, ydat, sigma=y_error)

    errors = np.sqrt(np.diag(pcov))
    yfit = tafel_equation(xdat, *popt)

    r2adj = calculate_r2(ydat, yfit, xdat.shape[0], len(popt))

    # plot fit
    ax.plot(
        xdat,
        yfit,
        label=f"Tafel fit\nslope = ({round(popt[1]*1e3, 1)} $\\pm$ {round(errors[1]*1e3, 1)}) $\\mathrm{{\\frac{{mV}}{{dec}} }}$ \n$R^2_\\mathrm{{adj.}}$= {round(r2adj, 4)}",
        ls="-",
        zorder=1,
    )
    ax.tick_params(direction="in", which="both")
    ax.legend(loc=0)

    for end in ["png", "svg", "pdf"]:
        f.savefig(f"{savename}_Tafel.{end}", dpi=300)

    plt.show()

    return [*popt, *errors]


def _analyze_chronopotentiometry(filename, header_num, area):
    """
    Plots chronopotentiometry data as Potential VS time, as well as current density VS time.
    The results is stored in a pd.DataFrame and as an Excel sheet.

    Parameters
    ----------
    filename : str, or path-like
        Name of the file. Should end with '.mpt'.
    area : int or float,
            active area of cell used in measurement in cm².

    Returns
    -------
    cp_df : pd.DataFrame
        The read-in DataFrame including basic SI conversions.

    """
    # load chronopotentiometry (cp) file as DataFrame
    # problem -> arbitrary header number close to header_num needed. Workaround to brute-force-finding the right one
    with open(filename) as file:
        rows = file.readlines()
    nrow_list = list(range(len(rows)))

    rotated_nrow_list = nrow_list[: header_num + 1][::-1] + nrow_list[header_num + 1 :]
    for i in rotated_nrow_list:
        try:
            cp_df = pd.read_csv(
                filename,
                header=i,
                sep="\t",  # causes artificial last column --> better Regex possible?
                decimal=",",
                encoding="latin",
            )
        except:
            cp_df = pd.DataFrame()
        if "mode" in cp_df.columns:
            print("Expected header lines:", header_num)
            print("Working header lines:", i)
            break
    # get rid of column
    try:
        cp_df.pop("Unnamed: 26")
    except KeyError:
        pass

    # calculate SI units
    cp_df = convert_raw_to_si(cp_df, area=area)

    # plot potential vs time

    f, ax = plt.subplots(figsize=(11, 5), layout="constrained", dpi=150)

    ax.plot("time/h", "<Ewe>/V", data=cp_df)
    ax.tick_params(
        direction="in",
        which="both",
    )
    ax.tick_params(color="C0", labelcolor="C0", axis="y")
    ax.set_ylabel("Potential / V", color="C0")
    ax.set_xlabel("Time / h")
    # get second y-Axis
    ax2 = ax.twinx()
    ax2.plot("time/h", "i/A*cm^-2", data=cp_df, color="C1", label="$j_\\mathrm{meas}$")
    ax2.plot("time/h", "i_control/A*cm^-2", "--", color="C1", data=cp_df, label="$j_\\mathrm{set}$")
    ax2.legend(loc=0)
    ax2.tick_params(direction="in", which="both")
    ax2.tick_params(color="C1", labelcolor="C1", axis="y")
    ax2.set_ylabel("Current density / (A cm$^{-2}$)", color="C1")

    savename = filename.split(".mpt")[0]

    for end in ["png", "svg", "pdf"]:
        f.savefig(f"{savename}_Chronopotentiometry.{end}", dpi=300)

    plt.show()
    plt.close("all")

    return cp_df


def _analyze_chronoamperometry(filename, header_num, area):
    """
    Plots chronopotentiometry data as Potential VS time, as well as current density VS time.
    The results is stored in a pd.DataFrame and as an Excel sheet.

    Parameters
    ----------
    filename : str, or path-like
        Name of the file. Should end with '.mpt'.
    area : int or float,
            active area of cell used in measurement in cm².

    Returns
    -------
    cp_df : pd.DataFrame
        The read-in DataFrame including basic SI conversions.

    """
    # load chronopotentiometry (cp) file as DataFrame
    # problem -> arbitrary header number close to header_num needed. Workaround to brute-force-finding the right one
    with open(filename) as file:
        rows = file.readlines()
    nrow_list = list(range(len(rows)))

    rotated_nrow_list = nrow_list[: header_num + 1][::-1] + nrow_list[header_num + 1 :]
    for i in rotated_nrow_list:
        try:
            cp_df = pd.read_csv(
                filename,
                header=i,
                sep="\t",  # causes artificial last column --> better Regex possible?
                decimal=",",
                encoding="latin",
            )
        except:
            cp_df = pd.DataFrame()
        if "mode" in cp_df.columns:
            print("Expected header lines:", header_num)
            print("Working header lines:", i)
            break
    # get rid of column
    try:
        cp_df.pop("Unnamed: 26")
    except KeyError:
        pass

    # calculate SI units
    cp_df = convert_raw_to_si(cp_df, area=area)

    # plot potential and current vs time

    f2, ax = plt.subplots(figsize=(11, 5), layout="constrained", dpi=150)

    ax.plot("time/h", "Ewe/V", data=cp_df)
    ax.tick_params(
        direction="in",
        which="both",
    )
    ax.tick_params(color="C0", labelcolor="C0", axis="y")
    ax.set_ylabel("Potential / V", color="C0")
    ax.set_xlabel("Time / h")
    # get second y-Axis
    ax2 = ax.twinx()
    ax2.plot("time/h", "i/A*cm^-2", data=cp_df, color="C1", label="$j_\\mathrm{meas}$")
    ax2.legend(loc=0)
    ax2.tick_params(direction="in", which="both")
    ax2.tick_params(color="C1", labelcolor="C1", axis="y")
    ax2.set_ylabel("Current density / (A cm$^{-2}$)", color="C1")
    # ax2.set_ylim((0, None))

    savename = filename.split(".mpt")[0]

    for end in ["png", "svg", "pdf"]:
        f2.savefig(f"{savename}_Chronoamperometry.{end}", dpi=300)

    plt.show()
    plt.close("all")
    return cp_df


def analyze_CP_data(file_name, last_second_interval, area):
    """
    High-level function to perform CP analysis on Biologic experimental data.

    Parameters
    ----------
    file_name : str
        name of the experimental data file in mpt format.
    last_second_interval : int, optional
        Time in seconds to average the signal over. The last last_second_interval seconds per cycle are regarded.
    area : int or float,
            active area of cell used in measurement in cm².

    Returns
    -------
    None.

    """
    header_rows = detect_header_lines(file_name)
    cp_df = _analyze_chronopotentiometry(file_name, header_num=header_rows, area=area)

    # split at half cycle and analyze each cycle separately
    usable_cycles = cp_df["half cycle"].unique()

    # get single cycle
    last_minute_dict_mean = {}
    last_minute_dict_std = {}

    # create sub dataframes for each half cycle to average
    for cycle in usable_cycles:
        sub_df = cp_df[cp_df["half cycle"] == cycle]
        max_time = sub_df["time/s"].max()
        last_minute_start = max_time - last_second_interval
        last_minute_cycle = sub_df[sub_df["time/s"] >= last_minute_start]
        # average last 60 s
        last_minute_dict_mean[cycle] = last_minute_cycle.mean()
        last_minute_dict_std[cycle] = last_minute_cycle.std(ddof=1)
    last_minute_df_mean = pd.DataFrame.from_dict(last_minute_dict_mean, orient="index")
    last_minute_df_std = pd.DataFrame.from_dict(last_minute_dict_std, orient="index")

    savename = file_name.split(".mpt")[0]
    # excel sheet with mean values
    last_minute_df_mean.to_excel(f"{savename}_mean_cycle_values.xlsx")
    # excel sheet with standard deviation
    last_minute_df_std.to_excel(f"{savename}_std_cycle_values.xlsx")

    # plot mean values with error over time
    f, ax = plt.subplots(figsize=(11, 5), layout="constrained", dpi=150)
    ax.errorbar(
        last_minute_df_mean["time/h"], last_minute_df_mean["<Ewe>/V"], yerr=last_minute_df_std["<Ewe>/V"], fmt="."
    )

    ax.tick_params(
        direction="in",
        which="both",
    )
    ax.tick_params(color="C0", labelcolor="C0", axis="y")
    ax.set_ylabel("Potential / V", color="C0")
    ax.set_xlabel("Time / h")
    # get second y-Axis
    ax2 = ax.twinx()
    ax2.plot(
        "time/h", "i/A*cm^-2", ".", data=last_minute_df_mean, color="C1", label="$j_\\mathrm{meas}$"
    )  #'i_control/A*cm^-2'
    ax2.plot("time/h", "i_control/A*cm^-2", "--", color="C1", data=cp_df, label="$j_\\mathrm{set}$")
    ax2.legend(loc=0)
    ax2.tick_params(direction="in", which="both")
    ax2.tick_params(color="C1", labelcolor="C1", axis="y")
    ax2.set_ylabel("Current density / (A cm$^{-2}$)", color="C1")
    for end in ["png", "svg"]:
        f.savefig(f"{savename}_CP_Overview.{end}", transparent=True)
    plt.show()
    plt.close("all")


def analyze_CA_data(file_name, last_second_interval, area):
    """
    High-level function to perform CA analysis on Biologic experimental data.

    Parameters
    ----------
    file_name : str
        File of the CA experiment in mpt format.
    last_second_interval : int, optional
        Amoutn of last seconds to average the signal over. The last last_second_interval seconds per cycle are regarded.
    Returns
    -------
    None.

    """
    header_rows = detect_header_lines(file_name)
    savename = file_name.split(".mpt")[0]
    ca_df = _analyze_chronoamperometry(file_name, header_num=header_rows, area=area)
    # split at cycle number and analyze each cycle separately
    usable_cycles = ca_df["cycle number"].unique()
    # get single cycle
    last_minute_dict_mean = {}
    last_minute_dict_std = {}
    time_duration = {}
    # create sub dataframes for each half cycle to average
    for cycle in usable_cycles:
        sub_df = ca_df[ca_df["cycle number"] == cycle]
        max_time = sub_df["time/s"].max()
        min_time = sub_df["time/s"].min()
        cycle_duration = max_time - min_time
        last_minute_start = max_time - last_second_interval
        last_minute_cycle = sub_df[sub_df["time/s"] >= last_minute_start]
        # average last 60 s
        last_minute_dict_mean[cycle] = last_minute_cycle.mean(skipna=True)
        last_minute_dict_std[cycle] = last_minute_cycle.std(ddof=1, skipna=True)
        time_duration[cycle] = cycle_duration

    last_minute_df_mean_ca = pd.DataFrame.from_dict(last_minute_dict_mean, orient="index")
    last_minute_df_std_ca = pd.DataFrame.from_dict(last_minute_dict_std, orient="index")

    f_duration, ax_duration = plt.subplots(figsize=(11, 5), layout="constrained", dpi=150)

    time_duration_keys = [key for key in time_duration]
    time_duration_values = [time_duration[key] for key in time_duration]
    ax_duration.plot(time_duration_keys, time_duration_values, ".")
    ax_duration.set(
        ylabel="Duration / s",
        xlabel="Cycle Number",
    )
    ax_duration.tick_params(
        direction="in",
        which="both",
    )
    for end in ["png", "svg"]:
        f_duration.savefig(f"{savename}_cycle_number_duration.{end}", transparent=True)

    plt.show()
    # save dataframe to excel
    savename = file_name.split(".mpt")[0]
    last_minute_df_mean_ca.to_excel(f"{savename}_mean_cycle_values.xlsx")
    last_minute_df_std_ca.to_excel(f"{savename}_std_cycle_values.xlsx")

    # plot CA overview
    f3, ax3 = plt.subplots(figsize=(11, 5), layout="constrained", dpi=150)
    ax3.errorbar(
        last_minute_df_mean_ca["time/h"], last_minute_df_mean_ca["Ewe/V"], yerr=last_minute_df_std_ca["Ewe/V"], fmt="."
    )

    ax3.tick_params(
        direction="in",
        which="both",
    )
    ax3.tick_params(color="C0", labelcolor="C0", axis="y")
    ax3.set_ylabel("Potential / V", color="C0")
    ax3.set_xlabel("Time / h")
    # get second y-Axis
    ax4 = ax3.twinx()
    ax4.plot(
        "time/h", "i/A*cm^-2", ".", data=last_minute_df_mean_ca, color="C1", label="$j_\\mathrm{meas}$"
    )  #'i_control/A*cm^-2'
    ax4.legend(loc=0)
    ax4.tick_params(direction="in", which="both")
    ax4.tick_params(color="C1", labelcolor="C1", axis="y")
    ax4.set_ylabel("Current density / (A cm$^{-2}$)", color="C1")
    for end in ["png", "svg"]:
        f3.savefig(f"{savename}_CA_overview.{end}", transparent=True)

    plt.show()
    plt.close("all")


def analyze_GEIS_data(
    file_name,
    steady_state_duration,
    tafel_run,
    numbers_per_repetition,
    frequency_interval,
    equivalent_circuit,
    initial_circuit_guess,
    hfr_element,
    area,
    tafel_plot_max_current_density,
    show_nyqust_plots=True,
):

    """
    High-level function to analyze galvanostatic EIS experiments of PEMWE cells using a BioLogic potentiostat.


    Parameters
    ----------
    file_name : str
        File of the GEIS experiment in mpt format.
    steady_state_duration : int
        Duration in seconds over which the steady state should be analyzed.
    tafel_run : int
        Run number (repetition) for which the Tafel analysis should be performed.
    numbers_per_repetition : int
        Number of current density steps per run (repetition).
    frequency_interval : tuple
        Tuple of two values containing two floating point values that denote the lower and upper
        frequency interval boundary for EIS fitting, (lower, upper).
    equivalent_circuit : str
        Electrical circuit notation matching impedance.py (see https://impedancepy.readthedocs.io/en/latest/getting-started.html#step-3-define-your-impedance-model).

        Note that 'TLMA' can be used to include a PEMWE-tailored transmission line model
        (see Makharia, Mathias and Baker, https://iopscience.iop.org/article/10.1149/1.1888367).

        For our work, 'L0-R0-TLMA0' was a good equivalent circuit estimate.
    initial_circuit_guess : list
        List of initial guess values as float or int to fit the chosen equivalent circuit to experimental data.
        This will be used for the first EIS fit. Subsequent fits use the first
        successful fit results as initial guess.
    hfr_element : str
        Element of the equivalent circuit that accounts for the high-frequency resistance. In our example, this would
        be 'R0'.
    area : float
        Active area of the single PEMWE cell to normalize the current values to.
    tafel_plot_max_current_density : float
        Upper current density threshold for Tafel analysis.
    show_nyqust_plots : bool, optional
        A flag defining whether nyquist fits should be plotted as intermediate steps. The default is True.

    Returns
    -------
    None.

    """
    save_name = file_name.split(".")[0]
    polcurve_df = polarization_curve(
        file_name,
        numbers_per_repetition=numbers_per_repetition,
        steady_state_length=steady_state_duration,
        area=area,
        savename=save_name,
    )
    nyquist_df = nyquist_plot(
        file_name,
        area,
        numbers_per_repetition=numbers_per_repetition,
        show=show_nyqust_plots,
        frequency_interval=frequency_interval,
        equivalent_circuit=equivalent_circuit,
        initial_circuit_guess=initial_circuit_guess,
    )
    hfr_df = hfr_evolution(polcurve_df, nyquist_df, savename=save_name, hfr_element=hfr_element, area=area)
    plot_hfr_evolution(hfr_df, savename=save_name)
    tafel_plot(
        hfr_df,
        run=tafel_run,
        savename=save_name,
        fit_limit=tafel_plot_max_current_density,
    )


def analyze_MB_data(
    file_name,
    steady_state_duration,
    tafel_run,
    numbers_per_repetition,
    frequency_interval,
    show_nyqust_plots,
    equivalent_circuit,
    initial_circuit_guess,
    hfr_element,
    area,
    tafel_plot_max_current_density,
):
    """
    High-level function to analyze Modulo Bat (MB) experiments consisting of current hold
    and EIS for PEMWE analysis using a BioLogic potentiostat.


    Parameters
    ----------
    file_name : str
        File of the MB experiment in mpt format.
    steady_state_duration : int
        Duration in seconds over which the steady state should be analyzed.
    tafel_run : int
         Run number (repetition) for which the Tafel analysis should be performed.
    numbers_per_repetition : int
        Number of current density steps per run (repetition).
    frequency_interval : tuple
        Tuple of two values containing two floating point values that denote the lower and upper
        frequency interval boundary for EIS fitting, (lower, upper).
    show_nyqust_plots : bool, optional
        A flag defining whether nyquist fits should be plotted as intermediate steps. The default is True.
    equivalent_circuit : str
        Electrical circuit notation matching impedance.py (see https://impedancepy.readthedocs.io/en/latest/getting-started.html#step-3-define-your-impedance-model).

        Note that 'TLMA' can be used to include a PEMWE-tailored transmission line model
        (see Makharia, Mathias and Baker, https://iopscience.iop.org/article/10.1149/1.1888367).

        For our work, 'L0-R0-TLMA0' was a good equivalent circuit estimate.
    initial_circuit_guess : list
        List of initial guess values as float or int to fit the chosen equivalent circuit to experimental data.
        This will be used for the first EIS fit. Subsequent fits use the first
        successful fit results as initial guess.
    hfr_element : str
        Element of the equivalent circuit that accounts for the high-frequency resistance. In our example, this would
        be 'R0'.
    area : float
        Active area of the single PEMWE cell to normalize the current values to.
    tafel_plot_max_current_density : float
        Upper current density threshold for Tafel analysis.

    Returns
    -------
    None.

    """
    save_name = file_name.split(".")[0]
    mb_df = provide_data(file_name, area)
    mb_df = mb_df[mb_df["i/A*cm^-2"] != 0]
    mb_df["old cycle number"] = mb_df["cycle number"]
    mb_df["cycle number"] = mb_df["half cycle"] + 1
    mb_df["<Ewe>/V"] = mb_df["Ewe/V"]

    polcurve_df = polarization_curve(
        mb_df,
        steady_state_length=steady_state_duration,
        numbers_per_repetition=numbers_per_repetition,
        savename=save_name,
        area=area,
    )
    nyquist_df = _nyquist_plot_core(
        mb_df,
        file_name,
        numbers_per_repetition=numbers_per_repetition,
        show=show_nyqust_plots,
        frequency_interval=frequency_interval,
        equivalent_circuit=equivalent_circuit,
        initial_circuit_guess=initial_circuit_guess,
    )

    hfr_df = hfr_evolution(polcurve_df, nyquist_df, savename=save_name, hfr_element=hfr_element, area=area)

    plot_hfr_evolution(hfr_df, savename=save_name)

    tafel_plot(hfr_df, run=tafel_run, fit_limit=tafel_plot_max_current_density, savename=save_name)


#%%
# actual execution
if __name__ == "__main__":

    # Global variables
    local_mtp_files = [file for file in os.listdir() if file.endswith(".mpt")]

    for file_name in local_mtp_files:

        print("\n")
        print("Starting with", file_name)
        print("\n")

        if re.search("CP_C\d+", file_name):
            analyze_CP_data(file_name, last_second_interval=60, area=5)

        elif re.search("CA_C\d+", file_name):
            analyze_CA_data(file_name, last_second_interval=60, area=5)

        elif re.search("GEIS_C\d+", file_name):
            analyze_GEIS_data(
                file_name,
                tafel_run=3,
                steady_state_duration=30,
                numbers_per_repetition=20,
                frequency_interval=(500, 5e4),
                equivalent_circuit="L0-R0-TLMA0",
                initial_circuit_guess=(0.00001, 0.01, 0.01, 0.1, 0.01),
                hfr_element="R0",
                area=5,
                tafel_plot_max_current_density=0.1,
                show_nyqust_plots=True,
            )

        elif re.search("MB_C\d+", file_name):
            analyze_MB_data(
                file_name,
                steady_state_duration=30,
                tafel_run=3,
                numbers_per_repetition=16,
                frequency_interval=(500, 5e4),
                show_nyqust_plots=True,
                equivalent_circuit="L0-R0-TLMA0",
                initial_circuit_guess=(0.00001, 0.01, 0.01, 0.1, 0.01),
                hfr_element="R0",
                area=5,
                tafel_plot_max_current_density=0.1,
            )

        else:
            print("Unknown measurement type. Ignoring", file_name)
