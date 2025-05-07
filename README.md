# ne571_p5

Consider the two-group three-dimensional nuclear reactor core simulator you developed in Project #2.  For this project, your program should allow you to model a reactor of arbitrary dimensions A, B, C.  Also, it should allow you substitute arbitrary materials in any discretized node or group of nodes.  So that you could assign regions with different fuel types, fuel with control rods, reflector/moderator materials, etc.

    1. Ensure that you can assign any set of properties to any numerical node in your model.  Provide evidence that you’ve benchmarked and validated your code so that it produces reasonable results (before you move to Item 2.).  This can be done with a homogeneous unreflected reactor.
    2. Create a representative model of a small modular reactor (SMR) by developing 2-group sets of cross sections employing CASMO or POLARIS and data from representative fuel assembly designs.  You have the choice of selecting the type of reactor you wish to model, but LWR technology is likely easier to model with fewer and shorter versions of typical LWR assemblies.
        a. Establish representative dimensions for the fuel assemblies (width and height), and the core (number of fuel assemblies in the core).  Note that you should probably use a sufficiently fine numerical mesh in the x, y, and z directions.
        b. Generate various discrete sets of two-group homogenized cross sections for your fuel assemblies by using CASMO or SCALE/Polaris.  Some examples are suggested below:
            i. Enrichment variation (3% to 5% U-235).  Consider HALEU at <20% U-235.
            ii. MOX fuel (various combinations of U-235 and Pu-239)
            iii. Burnup variation (0 to 45 GWD/MT)
            iv. Lattices with and without control rods
            v. Lattices with burnable absorbers
            vi. Develop Cross-Sectional Data for the reflector region on the outer part surrounding the core.
    3. Use your cross section sets from Item 2.b. to create a core design by creating an alternating/checkerboard arrangement of fresh, once/twice burnt, rodded/unrodded, and/or MOX fuel assemblies (in-out or low-leakage), with reflector regions above, below, and on the outermost radius of the core.
    4. Calculate k-effective for your reactor and perform modifications or changes to your design that can help bring it close to critical (so that k-effective=1.0000).
    5. Normalize your flux to an actual flux based on the total power of the reactor.  Plot and display average assembly flux and average nodal/assembly power, radially (X-Y) for each fuel assembly.  
    6. Develop the computational methodology assigned to your individual group (see below).
    7. Write a report, concise and to the point, and prepare a 20-minute presentation.  All members sign the report and assign a level of participation/contribution to the group based on a scale from 1 to 5, where 5 represents fully engaged and fully contributed, while 1 is the opposite.

Super Group Mario: Groups 1 and 2: Develop the computational methodology to evaluate the response of your reactor model to changes in fuel and moderator temperature. Establish a reasonable estimate for the temperature of the fuel and moderator (use projects 3b and 3c as the basis), generate appropriate cross sections, and compare results with temperature feedback to your base model which uses a constant average temperature for all nodes in the core.

*Super Group Luigi: Groups 3 and 4: Develop the computational methodology to adjust the boron concentration (or control rod insertion) in your simulator to automatically achieve a critical reactor.*

Extra Credit (10%): 
    a. Develop the computational methodology to adjust the burnup in fuel assemblies and to deplete the reactor from BOC to EOC.
    b. Automate the assignment of fuel properties to reduce power peaking or increase k-eff (or both). 
    
Link to overleaf report:
https://www.overleaf.com/6458736911wsbbnbkrhrcb#bd38bd

Link to presentation:
https://docs.google.com/presentation/d/1ZchmpGHHwa47pfYR_XyvaO_Z5YXXCLhl/edit?usp=sharing&ouid=111218421219816901221&rtpof=true&sd=true
