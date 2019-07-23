import copy

# input

def generateData(number_of_accounts = 50, number_of_group_transactions = 40, maximum_value = 100, seed = 0):
    """
    generate random data
    
    Returns:
        set: data
    """
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    names = []
    for i in range(number_of_accounts):
        names.append(chars[i % (len(chars)-1)])
    import numpy as np
    np.random.seed(seed)
    group_transactions = np.around(np.random.rand(number_of_group_transactions) * maximum_value, 2)
    group_transaction_account_ID = np.random.randint(0, number_of_accounts, number_of_group_transactions)

    account_weight = np.full(number_of_accounts, 1)
    group_transaction_includes_account = np.full((number_of_accounts,number_of_group_transactions), True)
    for i in range(1,number_of_accounts):
        if np.random.rand() > 0.5:
            account_weight[i] = 2
        for j in range(number_of_group_transactions):
            if np.random.rand() > 0.8:
                group_transaction_includes_account[i,j] = False
    return names, group_transactions, group_transaction_account_ID, group_transaction_includes_account, account_weight


def processData(names, group_transactions, group_transaction_account_ID, group_transaction_includes_account = None, account_weight = None, verbose = False):
    """
    Process the data to get the saldos.
    
    Args:
        names (list): list of names
        group_transactions (list): Original payments
        group_transaction_account_ID (list): Who payed?
        group_transaction_includes_account (ndarray): who has to 
        account_weight (ndarray): weighting of the accounts
        verbose (bool, optional): Defaults to False.

    Returns:
        list: rounded liability accounts balance
    """
    import numpy as np
    number_of_accounts = len(names)
    number_of_group_transactions = len(group_transactions)
    group_transaction_matrix = np.full((number_of_accounts, number_of_group_transactions), 0.0)
    for i in range(number_of_group_transactions):
        group_transaction_matrix[group_transaction_account_ID[i],i] = group_transactions[i]
    if (type(account_weight) != np.ndarray):
        account_weight = np.full(number_of_accounts, 1)
    if (type(group_transaction_includes_account) != np.ndarray):
        group_transaction_includes_account = np.full((number_of_accounts, number_of_group_transactions), True)

    # how much money has already been spent (decreases liabilities -> debit)
    liability_accounts_debit = group_transaction_matrix.sum(1)
    # how much money was spent in total?
    group_liability_balance = np.sum(group_transactions)

    # calculate the shares
    temp_planned_account_to_group_weight_matrix = (account_weight * group_transaction_includes_account.transpose()).transpose()
    account_to_group_weight_matrix = temp_planned_account_to_group_weight_matrix / temp_planned_account_to_group_weight_matrix.sum(0)
    # increase of liabilires
    liability_accounts_credit = np.dot(account_to_group_weight_matrix, group_transactions)
    # liability balance of each account
    liability_accounts_balance = liability_accounts_credit - liability_accounts_debit
    rounded_liability_accounts_balance = np.around(liability_accounts_balance, 2)

    if (verbose):
        for index, value in enumerate(liability_accounts_debit):
            print(names[index] + ' payed in total ' + str(value) + '€')
        for index, value in enumerate(liability_accounts_credit):
            print(names[index] + ' should have payed ' + str(value) + '€')
        for index, value in enumerate(rounded_liability_accounts_balance):
            print('The saldo of ' + names[index] + ' is ' + str(value) + '€')
    return rounded_liability_accounts_balance


def search_zero_sum_groups(rounded_liability_accounts_balance, initial_error = 0, check_level = 1, tolerance = 0, verbose = False):
    """
    Search groups of zero sum accounts.
    
    Args:
        rounded_liability_accounts_balance (list): balance of the liability accounts
        initial_error (float, optional): initial total error. Defaults to 0.
        check_level (int, optional): search complexity. Defaults to 1.
        tolerance (float, optional): accepted tolerance. Defaults to 0.
        verbose (bool, optional): Defaults to False.
    """

    positive_account_saldos = sorted([{'account ID' : i, 'account saldo' : rounded_liability_accounts_balance[i]} for i in range(len(rounded_liability_accounts_balance)) if rounded_liability_accounts_balance[i] > 0], key = lambda x : x['account saldo'], reverse=True)
    negative_account_saldos = sorted([{'account ID' : i, 'account saldo' : rounded_liability_accounts_balance[i]} for i in range(len(rounded_liability_accounts_balance)) if rounded_liability_accounts_balance[i] < 0], key = lambda x : x['account saldo'], reverse=True)
    groups = []
    positive_index_1 = 0
    total_error = initial_error
    while positive_index_1 < len(positive_account_saldos) and positive_index_1 >= 0:
        positive_account_saldo_1 = positive_account_saldos[positive_index_1]['account saldo']
        positive_account_ID_1 = positive_account_saldos[positive_index_1]['account ID']
        try:
            if (check_level > 0):
                negative_index_1 = 0
                while negative_index_1 < len(negative_account_saldos) and negative_index_1 >= 0:
                    negative_account_saldo_1 = negative_account_saldos[negative_index_1]['account saldo']
                    negative_account_ID_1 = negative_account_saldos[negative_index_1]['account ID']
                    if (abs(positive_account_saldo_1 + negative_account_saldo_1 + total_error/2) <= max(tolerance - abs(total_error), 0) + 1e-5):
                        # 1 - 1 group
                        if (verbose and abs(positive_account_saldo_1 + negative_account_saldo_1) > 0):
                            print (str(positive_account_saldo_1) + ' is equal to ' + str(negative_account_saldo_1))
                        groups.append(([positive_account_ID_1], [negative_account_ID_1]))
                        negative_account_saldos.pop(negative_index_1)
                        positive_account_saldos.pop(positive_index_1)
                        total_error += positive_account_saldo_1 + negative_account_saldo_1
                        raise RuntimeError()
                    if (check_level > 1):
                        negative_index_2 = 0
                        while negative_index_2 < len(negative_account_saldos) and negative_index_2 >= 0:
                            negative_account_saldo_2 = negative_account_saldos[negative_index_2]['account saldo']
                            negative_account_ID_2 = negative_account_saldos[negative_index_2]['account ID']
                            if negative_index_1 != negative_index_2:
                                if (abs(positive_account_saldo_1 + negative_account_saldo_1 + negative_account_saldo_2 + total_error/2) < max(tolerance - abs(total_error), 0) + 1e-5):
                                    # 1 - 2 group
                                    if (verbose and abs(positive_account_saldo_1 + negative_account_saldo_1 + negative_account_saldo_2) > 0):
                                        print (str(positive_account_saldo_1) + ' is equal to ' + str(negative_account_saldo_1 + negative_account_saldo_2))
                                    groups.append(([positive_account_ID_1], [negative_account_ID_1, negative_account_ID_2]))
                                    negative_account_saldos.pop(max(negative_index_1, negative_index_2))
                                    negative_account_saldos.pop(min(negative_index_1, negative_index_2))
                                    positive_account_saldos.pop(positive_index_1)
                                    total_error += positive_account_saldo_1 + negative_account_saldo_1 + negative_account_saldo_2
                                    raise RuntimeError()
                            negative_index_2 += 1
                    negative_index_1 += 1
                positive_index_2 = 0
                while positive_index_2 < len(positive_account_saldos) and positive_index_2 >= 0:
                    positive_account_saldo_2 = positive_account_saldos[positive_index_2]['account saldo']
                    positive_account_ID_2 = positive_account_saldos[positive_index_2]['account ID']
                    if (positive_index_2 != positive_index_1):
                        negative_index_1 = 0
                        while negative_index_1 < len(negative_account_saldos) and negative_index_1 >= 0:
                            negative_account_saldo_1 = negative_account_saldos[negative_index_1]['account saldo']
                            negative_account_ID_1 = negative_account_saldos[negative_index_1]['account ID']
                            if (abs(positive_account_saldo_1 + positive_account_saldo_2 + negative_account_saldo_1 + total_error/2) < max(tolerance - abs(total_error), 0) + 1e-5):
                                # 2 - 1 group
                                if (verbose and abs(positive_account_saldo_1 + positive_account_saldo_2 + negative_account_saldo_1 +  + total_error) > 0):
                                    print (str(positive_account_saldo_1 + positive_account_saldo_2) + ' is equal to ' + str(negative_account_saldo_1))
                                groups.append(([positive_account_ID_1, positive_account_ID_2], [negative_account_ID_1]))
                                negative_account_saldos.pop(negative_index_1)
                                positive_account_saldos.pop(max(positive_index_1, positive_index_2))
                                positive_account_saldos.pop(min(positive_index_1, positive_index_2))
                                total_error += positive_account_saldo_1 + positive_account_saldo_2 + negative_account_saldo_1
                                raise RuntimeError()
                            if (check_level > 2):
                                negative_index_2 = 0
                                while negative_index_2 < len(negative_account_saldos) and negative_index_2 >= 0:
                                    # 2 - 2 group
                                    negative_account_saldo_2 = negative_account_saldos[negative_index_2]['account saldo']
                                    negative_account_ID_2 = negative_account_saldos[negative_index_2]['account ID']
                                    if negative_index_1 != negative_index_2:
                                        if (abs(positive_account_saldo_1 + positive_account_saldo_2 + negative_account_saldo_1 + negative_account_saldo_2 + total_error/2) < max(tolerance - abs(total_error), 0) + 1e-5):
                                            if (verbose and abs(positive_account_saldo_1 + positive_account_saldo_2 + negative_account_saldo_1 + negative_account_saldo_2) > 0):
                                                print (str(positive_account_saldo_1 + positive_account_saldo_2) + ' is equal to ' + str(negative_account_saldo_1 + negative_account_saldo_2))
                                            groups.append(([positive_account_ID_1, positive_account_ID_2], [negative_account_ID_1, negative_account_ID_2]))
                                            negative_account_saldos.pop(max(negative_index_1, negative_index_2))
                                            negative_account_saldos.pop(min(negative_index_1, negative_index_2))
                                            positive_account_saldos.pop(max(positive_index_1, positive_index_2))
                                            positive_account_saldos.pop(min(positive_index_1, positive_index_2))
                                            total_error += positive_account_saldo_1 + positive_account_saldo_2 + negative_account_saldo_1 + negative_account_saldo_2
                                            raise RuntimeError()
                                    negative_index_2 += 1
                            negative_index_1 += 1
                    positive_index_2 += 1
        except:
            positive_index_1 -= 1
            pass
        positive_index_1 += 1
    return groups, total_error
           
def transaction_model_A(unbalanced_liability_accounts):
    """
    This model selects source and target for a transaction as follows:
    The highest negative saldo and the highest positive saldo which is positive are selected.
    
    Args:
        unbalanced_liability_accounts (list): Sorted list of unbalanced accounts.

    Returns:
        (int, int): source_index, target_index
    """
    # select highest debit value
    source_index = 0
    # select highest credit value
    target_index = 1
    while (target_index < len(unbalanced_liability_accounts) and unbalanced_liability_accounts[target_index]['account saldo'] >= 0):
        target_index += 1
    target_index = max(1, min(len(unbalanced_liability_accounts) - 1, target_index))
    source_index = target_index - 1    

    # use minimum value of source and target saldo
    if (-unbalanced_liability_accounts[source_index]['account saldo'] <= unbalanced_liability_accounts[target_index]['account saldo']):
        transaction_amount = -unbalanced_liability_accounts[target_index]['account saldo']
        remove_account_index = target_index
    else:
        transaction_amount = unbalanced_liability_accounts[source_index]['account saldo']
        remove_account_index = source_index
    return source_index, target_index, remove_account_index, transaction_amount



def transaction_model_B(unbalanced_liability_accounts):
    """
    This model selects source and target for a transaction as follows:
    The highest negative saldo and the highest positive are selected.
    Money is transferred from the liability account with the highest credit to the account with the highest debit value.
    The amount is the minimum of the credit and debit value.
    
    Args:
        unbalanced_liability_accounts (list): Sorted list of unbalanced accounts.

    Returns:
        (int, int): source_index, target_index
    """
    # select highest credit value
    source_index = 0
    # select highest debit value
    target_index = -1

    # use minimum value of source and target saldo
    if (-unbalanced_liability_accounts[source_index]['account saldo'] <= unbalanced_liability_accounts[target_index]['account saldo']):
        transaction_amount = -unbalanced_liability_accounts[target_index]['account saldo']
        remove_account_index = target_index
    else:
        transaction_amount = unbalanced_liability_accounts[source_index]['account saldo']
        remove_account_index = source_index
    return source_index, target_index, remove_account_index, transaction_amount

def transaction_model_C(unbalanced_liability_accounts):
    """
    This model selects source and target for a transaction as follows:
    Money is transferred from the liability account with the highest credit to the account with the highest debit value 
    which smaller than the credit value. The amount is the maximum of the credit and debit value.
    
    Args:
        unbalanced_liability_accounts (list): Sorted list of unbalanced accounts.

    Returns:
        (int, int): source_index, target_index
    """
    # select highest liability credit
    source_index = 0
    # select highest liability debit which is smaller than the liability credit
    target_index = 1
    while (target_index < len(unbalanced_liability_accounts) and unbalanced_liability_accounts[target_index]['account saldo'] >= -unbalanced_liability_accounts[source_index]['account saldo']):
        target_index += 1
    target_index = max(1, min(len(unbalanced_liability_accounts) - 1, target_index - 1))

    # use maximum value of source and target saldo
    if (-unbalanced_liability_accounts[source_index]['account saldo'] >= unbalanced_liability_accounts[target_index]['account saldo']):
        transaction_amount = -unbalanced_liability_accounts[target_index]['account saldo']
        remove_account_index = target_index
    else:
        transaction_amount = unbalanced_liability_accounts[source_index]['account saldo']
        remove_account_index = source_index
    return source_index, target_index, remove_account_index, transaction_amount

def group_and_binary_model_solver(rounded_liability_accounts_balance, check_level = 2, tolerance = 0, transaction_model = transaction_model_C, verbose = False):
    unbalanced_liability_accounts = sorted([{'account ID' : i, 'account saldo' : rounded_liability_accounts_balance[i]} for i in range(len(rounded_liability_accounts_balance))], key = lambda x : x['account saldo'], reverse=True)
    transaction_vector = []

    total_error = 0
    groups, total_error = search_zero_sum_groups([unbalanced_liability_accounts[i]['account saldo'] for i in range(len(unbalanced_liability_accounts))], total_error, check_level, tolerance)
    delete_accounts = set()
    for member in groups:
        if (len(member[0]) == 1 and len(member[1]) == 1):
            delete_accounts = delete_accounts.union(member[0])
            delete_accounts = delete_accounts.union(member[1])
            transaction_vector.append((unbalanced_liability_accounts[member[0][0]]['account ID'], unbalanced_liability_accounts[member[1][0]]['account ID'], unbalanced_liability_accounts[member[0][0]]['account saldo']))
        else:
            group_member_ids = member[0] + member[1]
            group_member_values = [unbalanced_liability_accounts[i]['account saldo'] for i in group_member_ids]
            group_transaction_vector = binary_model_solver(group_member_values, check_level, tolerance, transaction_model, verbose)
            for i in range(len(group_transaction_vector)):
                group_transaction_vector[i] = (unbalanced_liability_accounts[group_member_ids[group_transaction_vector[i][0]]]['account ID'], unbalanced_liability_accounts[group_member_ids[group_transaction_vector[i][1]]]['account ID'], group_transaction_vector[i][2])
            delete_accounts = delete_accounts.union(member[0])
            delete_accounts = delete_accounts.union(member[1])
            transaction_vector += group_transaction_vector
    for index in sorted(list(delete_accounts), reverse=True):
        unbalanced_liability_accounts.pop(index)
    remaining_values = [account['account saldo'] for account in unbalanced_liability_accounts]
    remaining_transaction_vector = binary_model_solver(remaining_values, check_level, tolerance, transaction_model, verbose)
    for i in range(len(remaining_transaction_vector)):
        remaining_transaction_vector[i] = (unbalanced_liability_accounts[remaining_transaction_vector[i][0]]['account ID'], unbalanced_liability_accounts[remaining_transaction_vector[i][1]]['account ID'], remaining_transaction_vector[i][2])
    transaction_vector += remaining_transaction_vector
    return transaction_vector

def serialize_nested_lists(input_value):
    if (type(input_value)==list and type(input_value[0])==int):
        return input_value
    else:
        value = []
        for item in input_value:
            value += serialize_nested_lists(item)
        return value

def binary_model_solver(rounded_liability_accounts_balance, check_level = 2, tolerance = 5, transaction_model = transaction_model_C, verbose = False):
    """
    This model balances accounts starting with the largest debt. Minimize the number of transactions per account.
    
    Args:
        rounded_liability_accounts_balance (ndarray): rounded balance for each liability account.
        verbose (bool): Activate debug messages. defaults to False.
    
    Returns:
        list: transactions
    """        
    unbalanced_liability_accounts = sorted([{'account ID' : i, 'account saldo' : rounded_liability_accounts_balance[i]} for i in range(len(rounded_liability_accounts_balance))], key = lambda x : x['account saldo'], reverse=True)
    transaction_vector = []
    
    while (len(unbalanced_liability_accounts)>1):
        unbalanced_liability_accounts = sorted(unbalanced_liability_accounts, key = lambda x : x['account saldo'], reverse=True)
        source_index, target_index, remove_account_index, transaction_amount = transaction_model(unbalanced_liability_accounts)
        source_account_ID = unbalanced_liability_accounts[source_index]['account ID']
        target_account_ID = unbalanced_liability_accounts[target_index]['account ID']
        
        #update accounts
        if (verbose):
            print(str(unbalanced_liability_accounts))
        unbalanced_liability_accounts[source_index]['account saldo'] = unbalanced_liability_accounts[source_index]['account saldo'] - transaction_amount
        unbalanced_liability_accounts[target_index]['account saldo'] = unbalanced_liability_accounts[target_index]['account saldo'] + transaction_amount
        if (verbose):
            print(str(unbalanced_liability_accounts))
        transaction_vector.append((source_account_ID, target_account_ID, transaction_amount))

        #remove balanced accounts
        unbalanced_liability_accounts.pop(remove_account_index)
    return transaction_vector
