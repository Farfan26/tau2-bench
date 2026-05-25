def retail_farfan_get_environment():
    """
    Returns the environment/context for the retail_farfan domain.
    This is required by the tau2 registry system.
    """
    from tau2.domains.retail_farfan.data_model import RetailDB
    from tau2.domains.retail_farfan.tools import RetailTools

    # Initialize with empty/default database
    db = RetailDB(
        users={},
        products={},
        orders={},
        returns={},
        payments={},
    )
    return {
        "db": db,
        "tools": RetailTools(db),
    }
