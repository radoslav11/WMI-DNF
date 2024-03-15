import numpy as np
from math import ceil


def generateDNF(
    nbVars,
    nbClauses,
    minCWidth,
    maxCWidth,
    R=0,
    Q=1,
    uniformisePrivilegedVars=True,
):
    # Min appearances ensures that all variables are used
    # Step 1: Generate widths of all clauses
    # Removed hard width requirement
    nbClauseSlots = 0
    clauseWidths = []
    failedAttempts = 0
    while nbClauseSlots < nbVars:
        clauseWidths = (
            np.random.randint(maxCWidth - minCWidth + 1, size=nbClauses)
            + minCWidth
        )
        # Minimum Appearances is Hard-Coded as 1. Balls (appearances) in boxes (variables)
        nbClauseSlots = np.sum(clauseWidths)
        if nbClauseSlots < nbVars:
            failedAttempts += 1
            if failedAttempts >= 5:
                raise TypeError(
                    " Could not generate sufficient slots in 5 different"
                    " attempts. Try a more generous allocation"
                )
    clauseVacancies = np.copy(clauseWidths)
    # New: Introduction of R and Q
    excess = nbClauseSlots - nbVars  # How many extra slots do we have?
    excessAllocation = int(R * excess)
    if Q > 0 and R > 0:
        standardAllocation = (
            nbClauseSlots - excessAllocation
        )  # Equivalent to nbVars + (1 - R)*excess
    else:
        standardAllocation = nbClauseSlots  # As before
    nbSeparators = standardAllocation - 1
    separators = (
        np.random.choice(nbSeparators, size=nbVars - 1, replace=False) + 1
    )  # Allocate such that all variables get at least one spot
    separators.sort()
    separators = np.append(separators, standardAllocation)
    separators = np.insert(separators, 0, 0)
    varAllocations = np.diff(separators)
    if Q > 0 and R > 0:
        nbPrivilegedVars = ceil(
            Q * nbVars
        )  # These are the variables that will be allocated additional appearances.
        #  Made Ceil to be more fair if Q is randomised
        privilegedVars = np.random.choice(
            nbVars, size=nbPrivilegedVars, replace=False
        )  # Choose "privileged" vars
        # Now allocate balls in boxes (with empty boxes allowed)
        privilegedSeparators = np.random.choice(
            excessAllocation + nbPrivilegedVars - 1,
            size=nbPrivilegedVars - 1,
            replace=False,
        )
        privilegedSeparators.sort()
        privilegedSeparators = np.append(
            privilegedSeparators, excessAllocation + nbPrivilegedVars - 1
        )  # Bug fix: Forgot - 1
        privilegedSeparators = np.insert(
            privilegedSeparators, 0, -1
        )  # To avoid need for an indexed call
        privilegedVarAllocations = (
            np.diff(privilegedSeparators) - 1
        )  # Almost similar to the non-empty case, except differences have to be subbed by one
        varAllocations[privilegedVars] += privilegedVarAllocations
    # End of new part
    clauseIndices = np.arange(nbClauses)
    runningSum = 0
    lastIndex = 0
    variableOrder = np.argsort(varAllocations)[::-1]  # Decreasing order
    # To deal with non-uniformities... fill them first
    maxVacancy = np.max(
        clauseVacancies
    )  # Initialise the counting of "maximally vacant" clauses. Initially = maxWidth
    clausesExactlyAtMaxVacancy = clauseVacancies == maxVacancy
    clausesOverMaxVacancy = clauseVacancies > maxVacancy
    clausesGEMaxVacancy = np.logical_or(
        clausesExactlyAtMaxVacancy, clausesOverMaxVacancy
    )
    maxVacancyClauseCount = np.count_nonzero(clausesGEMaxVacancy)
    clauses = np.full(
        (nbClauses, maxVacancy), 2 * nbVars
    )  # Use a uniformised representation
    for oIndex, index in enumerate(variableOrder):
        rerun = True  # This is to handle the exceptional case where we exceed the clauseCount at the final index (Mutex break)
        while rerun:
            allocation = varAllocations[index]
            if (
                runningSum + allocation > maxVacancyClauseCount
                or oIndex == nbVars - 1
            ):  # We have reached the threshold or end of list
                if (
                    oIndex == nbVars - 1
                ):  # Jan 29 Note: These are not Mutex --> This is a bug. What if both conditions are met at the same time?
                    relevantAllocations = varAllocations[
                        variableOrder[lastIndex:]
                    ]  # Step 1: Extract the relevant variable allocations
                    relevantVariables = variableOrder[
                        np.arange(lastIndex, nbVars)
                    ]  # 1.1 get the indices
                    if runningSum + allocation <= maxVacancyClauseCount:
                        runningSum += allocation  # Use the last variable
                        rerun = False
                    else:  # only use this to disable rerun
                        rerun = True  # This is where we need a rerun
                        relevantAllocations = varAllocations[
                            variableOrder[lastIndex:oIndex]
                        ]  # 1
                        relevantVariables = variableOrder[
                            np.arange(lastIndex, oIndex)
                        ]  # 1.1
                else:
                    rerun = False
                    relevantAllocations = varAllocations[
                        variableOrder[lastIndex:oIndex]
                    ]  # 1
                    relevantVariables = variableOrder[
                        np.arange(lastIndex, oIndex)
                    ]  # 1.1
                    # Slight Bug Fix to allow more randomised distribution across variables
                variableAllocsSeparate = np.repeat(
                    relevantVariables, relevantAllocations
                )  # 1.2 translate from multiset to list
                exactClauses = clauseIndices[
                    clausesExactlyAtMaxVacancy
                ]  # Step 2: Compute Clauses at and/or over max vacancy
                overClauses = clauseIndices[clausesOverMaxVacancy]
                nbOverClauses = overClauses.shape[0]
                selectedClauses = np.zeros(runningSum, int)
                # Heuristic : Always select clauses STRICTLY OVER max vacancy benchmark
                dropMaxVacancy = True  # Default is False, becomes True if more > max variables ,in which case next iter should have same max
                if nbOverClauses > runningSum:
                    selectedClauses = np.random.choice(
                        overClauses, size=runningSum, replace=False
                    )  # Randomly pick from the overs
                    dropMaxVacancy = False
                    # This will reduce their numbers but some will survive. In this case, max Vacancy must not drop
                else:
                    selectedClauses[:nbOverClauses] = overClauses
                    """print(len(exactClauses))
                    print(nbOverClauses)
                    print(runningSum)
                    print(allocation)
                    print("------------")"""
                    selectedClauses[nbOverClauses:] = np.random.choice(
                        exactClauses,
                        size=runningSum - nbOverClauses,
                        replace=False,
                    )  # Step 3: Select "runningSum" clauses
                """  This is not a perfect process. Some maximally vacant clauses will not be filled. So, to avoid having 
                the next iteration trying to fill the max again, which would be 1 or 2 clauses, which could end up thrashing
                indefinitely, we loosen the def of maxVacancy to a decrementing running count, which decrements at every 
                draw and then sampling is done at >= to max, so that omitted clauses are caught up with later """
                # Feb 1 Addition: Not uniform literal selection when Q, R > 0
                if (
                    uniformisePrivilegedVars and Q > 0 and R > 0
                ):  # New February 1
                    # Make all literals corresponding to a privileged variable all positive or negative
                    privilegedVariables = np.isin(
                        variableAllocsSeparate, privilegedVars
                    )  # Identify privileged vars in current alloc
                    # Because of condition on Q and R, privilegedVars will be initialised, so no worry there...
                    nonPrivilegedVariables = np.logical_not(
                        privilegedVariables
                    )  # And the non-privileged
                    nbOfNonPrivVar = np.sum(1 * nonPrivilegedVariables)
                    # To separate privileged and non-privileged in the original alloc array, "negate" the privileged.
                    separationVarAlloc = (
                        nbVars * privilegedVariables + variableAllocsSeparate
                    )
                    uniqueValues, indices = np.unique(
                        separationVarAlloc, return_inverse=True
                    )
                    uniquePrivMask = uniqueValues >= nbVars
                    uniquePrivVars = uniqueValues[
                        uniquePrivMask
                    ]  # Now identify the unique privileged variables
                    nbPrivVarsInAlloc = uniquePrivVars.shape[0]
                    if nbPrivVarsInAlloc > 0:
                        uniformRandomisation = (
                            np.random.randint(2, size=nbPrivVarsInAlloc)
                            * nbVars
                        )
                        # And the punch line ... update the new unique values with the random values
                        uniqueValues[uniquePrivMask] = (
                            uniqueValues[uniquePrivMask] % nbVars
                            + uniformRandomisation
                        )
                        literalAllocsSeparate = uniqueValues[
                            indices
                        ]  # Privileged vars treated
                        # And now allocate the non-privileged variables as usual
                        literalAllocsSeparate[nonPrivilegedVariables] += (
                            np.random.randint(2, size=nbOfNonPrivVar) * nbVars
                        )
                    else:  # No Privileged Variables. Treat as before
                        literalAllocsSeparate = (
                            variableAllocsSeparate
                            + np.random.randint(2, size=runningSum) * nbVars
                        )
                else:
                    literalAllocsSeparate = (
                        variableAllocsSeparate
                        + np.random.randint(2, size=runningSum) * nbVars
                    )  # Step 4: Compute Literal Negation
                clauses[
                    selectedClauses, clauseVacancies[selectedClauses] - 1
                ] = literalAllocsSeparate  # Step 5: Use Advanced Indexing to update clauses
                """ Writing into the clause array is happening in the reverse order to avoid unnecessary subtraction of 
                clause widths at every time step """
                lastIndex = oIndex  # Step 6: Update starting point
                clauseVacancies[
                    selectedClauses
                ] -= 1  # Step 7: Update the selected clauses' fullness
                if not rerun:
                    runningSum = allocation  # Step 8: Reset the running sum to the starting value
                else:
                    runningSum = (
                        0  # So as not to count the last element twice.
                    )
                    # Normally it increments to the one after, but in the rerun case it's the same number that would be counted twice
                # Step 9: Update the max vacancy and the rest now
                if (
                    dropMaxVacancy
                ):  # Only do this if nb > max <= runningSum, otherwise we'll need to run again at the same max
                    maxVacancy = (
                        maxVacancy - 1 if maxVacancy > 1 else maxVacancy
                    )  # Loose update based on comment above. Don't allow 0 max vacancy otherwise can fill full clauses
                clausesExactlyAtMaxVacancy = clauseVacancies == maxVacancy
                clausesOverMaxVacancy = clauseVacancies > maxVacancy
                clausesGEMaxVacancy = np.logical_or(
                    clausesExactlyAtMaxVacancy, clausesOverMaxVacancy
                )
                maxVacancyClauseCount = np.count_nonzero(clausesGEMaxVacancy)
                # Step 10: If the new max is still smaller than the new running sum, shrink further
                while (
                    runningSum > maxVacancyClauseCount
                    and dropMaxVacancy
                    and not oIndex == nbVars - 1
                ):  # Only matters if you're not at the end
                    if maxVacancy > 1:
                        maxVacancy -= 1
                        # This is a bit more tedious ... but is worth it considering the algorithm is less likely to crash
                        clausesExactlyAtMaxVacancy = (
                            clauseVacancies == maxVacancy
                        )
                        clausesOverMaxVacancy = clauseVacancies > maxVacancy
                        clausesGEMaxVacancy = np.logical_or(
                            clausesExactlyAtMaxVacancy, clausesOverMaxVacancy
                        )
                        maxVacancyClauseCount = np.count_nonzero(
                            clausesGEMaxVacancy
                        )
                    else:
                        raise TypeError(
                            "Generation Failed. Please Try again"
                        )  # You've reached a point where a single var has
                        # more allocations than the whole number of remaining clauses. Declare failure.
            else:
                rerun = False
                runningSum += allocation
    clausesPost = clauses[clauses < 2 * nbVars]
    splitIndeces = np.cumsum(clauseWidths)
    clausesNonUnif = np.split(clausesPost, splitIndeces)[
        :-1
    ]  # Result of last op is empty (This is a very slow operation)
    return clausesNonUnif  # The individual clauses have to be numpy arrays


def printDNF(conjunctions, nbVars):
    for conjunction in conjunctions:
        print("(", end=" ")
        for index, value in enumerate(conjunction):
            negated = (value / nbVars) >= 1
            if negated:
                print("~", end=" ")
            print("x_" + str(value % nbVars), end=" ")
            if index < len(conjunction) - 1:
                print("^", end=" ")
        print(")OR", end="")
    print()
