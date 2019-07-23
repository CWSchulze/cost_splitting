import pytest
import cost_splitting
import copy


def printStatistics (names, transaction_vector, printtransactions=False):
    if (printtransactions):
        for source, target, amount in transaction_vector:
            print ('{Source} has to pay {Amount:6.2f}€ to {Target}'.format(Source = names[source], Target = names[target], Amount = round(amount,2)))

    targets = set([target for source, target, amount in transaction_vector])
    sources = set([source for source, target, amount in transaction_vector])
    print ('Total number of transactions: {noOftransactions}'.format(noOftransactions = len(transaction_vector)) )
    print ('Total amount of money moved: {movedMoney}'.format(movedMoney = sum([amount for source, target, amount in transaction_vector])))
    print ('Targets: {targets} (total number = {numberOfTargets})'.format(targets = ", ".join([names[t] for t in targets]), numberOfTargets = len(targets)))
    print ('Sources: {sources} (total number = {numberOfTargets})'.format(sources = ", ".join([names[s] for s in sources]), numberOfTargets = len(sources)))

def print_transactions_matrix(names, transaction_vector):
    import numpy as np
    matrix = np.full((len(names), len(transaction_vector)), 0.0)
    for index, (source, target, amount) in enumerate(transaction_vector):
        matrix[source, index] = -amount
        matrix[target, index] = amount
    print (str(matrix))

def calculate_remaining_saldo(rounded_liability_accounts, transaction_vector, tolerance = 0, verbose = False):
    tolerance = max(tolerance, len(rounded_liability_accounts)*0.01)
    testData = copy.copy(rounded_liability_accounts)
    for source, target, amount in transaction_vector:
        testData[source] -= amount
        testData[target] += amount
    error = max([abs(data) for data in testData]) 
    print ('The error is ' + str(error))
    return error > tolerance, testData

def case_one_receiver_test(number_of_accounts = 20, value = 10.0):
    """
    Exemplay data. Several payers, one receiver.
    """
    
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    names = []
    for i in range(number_of_accounts):
        names.append(chars[i % (len(chars)-1)])

    rounded_liability_accounts = [round(-value, 2)] * (number_of_accounts - 1)
    sum_of_values = sum(rounded_liability_accounts)
    rounded_liability_accounts += [-sum_of_values]
    
    errorExceeded, liability_accounts_balances = calculate_remaining_saldo(rounded_liability_accounts, [(i, i+1, sum(rounded_liability_accounts[0:i+1])) for i in range(number_of_accounts-1)], tolerance = 5)
    if (errorExceeded):
        raise pytest.fail("Error too large. Liability accounts balances: "+str(liability_accounts_balances))
    
    return names, rounded_liability_accounts

def case_one_payer_test(number_of_accounts = 20, value = 10.0):
    """
    Exemplay data. Several payers, one payer.
    """
    
    names, rounded_liability_accounts = case_one_receiver_test(number_of_accounts, value)
    rounded_liability_accounts = [-saldo for saldo in rounded_liability_accounts]
    
    errorExceeded, liability_accounts_balances = calculate_remaining_saldo(rounded_liability_accounts, [(i, i+1, sum(rounded_liability_accounts[0:i+1])) for i in range(number_of_accounts-1)], tolerance = 5)
    if (errorExceeded):
        raise pytest.fail("Error too large. Liability accounts balances: "+str(liability_accounts_balances))
    return names, rounded_liability_accounts

def case_tolerance_test(value = 5.0, tolerance = 5.0):
    """
    Exemplay data. Several payers, one payer.
    """
    
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    f = 2
    group_A = [round(value*f + value*5*(i), 2) for i in range(1)]
    group_A = group_A + [0.9 * tolerance - sum(group_A)]
    f = 8
    group_B = [round(value*f + value*5*i, 2) for i in range(2)]
    group_B = group_B + [0.9 * tolerance - sum(group_B)]
    f = 64
    group_C = [-round(value*f + value*5*i, 2) for i in range(2)]
    group_C = group_C + [- 0.9 * tolerance - sum(group_C)]
    f = 512
    group_D = [round(value*f + value*5*(i-0.5), 2) for i in range(2)] + [-round(value*f + value*10*(i-0.5) + 0.9*tolerance/2, 2) for i in range(2)]
    f = 1024
    group_E = [-round(value*f + value*5*i, 2) for i in range(3)]
    group_E = group_E + [- sum(group_E)]

    rounded_liability_accounts = group_A + group_B + group_C + group_D + group_E
    number_of_accounts = len(rounded_liability_accounts)
    names = []
    for i in range(number_of_accounts):
        names.append(chars[i % (len(chars)-1)])
    
    errorExceeded, liability_accounts_balances = calculate_remaining_saldo(rounded_liability_accounts, [(i, i+1, sum(rounded_liability_accounts[0:i+1])) for i in range(number_of_accounts-1)], tolerance = tolerance)
    if (errorExceeded):
        raise pytest.fail("Error too large. Liability accounts balances: "+str(liability_accounts_balances))
    
    return names, rounded_liability_accounts

def model_test_sequence(check_level = 1, tolerance = 0, model = cost_splitting.transaction_model_A):
    names, group_transactions, group_transaction_account_ID, group_transaction_includes_account, account_weight = cost_splitting.generateData()
    rounded_liability_accounts = cost_splitting.processData(names, group_transactions, group_transaction_account_ID, group_transaction_includes_account, account_weight)
    transaction_vector = cost_splitting.group_and_binary_model_solver(rounded_liability_accounts, check_level=check_level, transaction_model = cost_splitting.transaction_model_B, verbose=False)
    printStatistics(names, transaction_vector, False)
    errorExceeded, liability_accounts_balances = calculate_remaining_saldo(rounded_liability_accounts, transaction_vector)
    if (errorExceeded):
        raise pytest.fail("Error too large. Liability accounts balances: "+str(liability_accounts_balances))

def test_model_with_tolerance_A():
    model_test_sequence(tolerance = 5, model = cost_splitting.transaction_model_A)
def test_model_with_tolerance_B():
    model_test_sequence(tolerance = 5, model = cost_splitting.transaction_model_B)
def test_model_with_tolerance_C():
    model_test_sequence(tolerance = 5, model = cost_splitting.transaction_model_C)

def test_model_A():
    model_test_sequence(model = cost_splitting.transaction_model_A)
def test_model_B():
    model_test_sequence(model = cost_splitting.transaction_model_B)
def test_model_C():
    model_test_sequence(model = cost_splitting.transaction_model_C)

def test_model_no_groups_A():
    model_test_sequence(check_level = 0, model = cost_splitting.transaction_model_A)
def test_model_no_groups_B():
    model_test_sequence(check_level = 0, model = cost_splitting.transaction_model_B)
def test_model_no_groups_C():
    model_test_sequence(check_level = 0, model = cost_splitting.transaction_model_C)

def test_model_C_one_receiver():
    names, rounded_liability_accounts = case_one_receiver_test()
    transaction_vector = cost_splitting.group_and_binary_model_solver(rounded_liability_accounts)#, check_level=0, tolerance=0, verbose=False)
    printStatistics(names, transaction_vector, False)
    errorExceeded, liability_accounts_balances = calculate_remaining_saldo(rounded_liability_accounts, transaction_vector)
    if (errorExceeded):
        raise pytest.fail("Error too large. Liability accounts balances: "+str(liability_accounts_balances))

def test_model_C_one_payer():
    names, rounded_liability_accounts = case_one_payer_test()
    transaction_vector = cost_splitting.group_and_binary_model_solver(rounded_liability_accounts)#, check_level=0, tolerance=0, verbose=False)
    printStatistics(names, transaction_vector, False)
    errorExceeded, liability_accounts_balances = calculate_remaining_saldo(rounded_liability_accounts, transaction_vector)
    if (errorExceeded):
        raise pytest.fail("Error too large. Liability accounts balances: "+str(liability_accounts_balances))

def test_model_C_tolerance():
    names, rounded_liability_accounts = case_tolerance_test()
    transaction_vector = cost_splitting.group_and_binary_model_solver(rounded_liability_accounts, tolerance = 5, verbose=False)
    printStatistics(names, transaction_vector, False)
    errorExceeded, liability_accounts_balances = calculate_remaining_saldo(rounded_liability_accounts, transaction_vector, tolerance = 5, verbose = True)
    if (errorExceeded):
        raise pytest.fail("Error too large. Liability accounts balances: "+str(liability_accounts_balances))
    names, rounded_liability_accounts = case_tolerance_test(tolerance=0)
    transaction_vector = cost_splitting.group_and_binary_model_solver(rounded_liability_accounts, tolerance = 0.0, verbose=False)
    printStatistics(names, transaction_vector, False)
    errorExceeded, liability_accounts_balances = calculate_remaining_saldo(rounded_liability_accounts, transaction_vector, tolerance = 0.0, verbose = True)
    if (errorExceeded):
        raise pytest.fail("Error too large. Liability accounts balances: "+str(liability_accounts_balances))


