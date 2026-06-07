"""Watershed and basin context page."""

import streamlit as st

from hydrology.app.shared import get_inventory
from hydrology.app.page_modules.advanced import _watershed_view


def show():
    """Render watershed inventory, basin boundaries, and basin characteristics."""
    st.header("Watershed")
    st.caption("Inspect basin boundaries, dams, land cover, and HUC inventory context.")

    inventory_df = get_inventory()
    if inventory_df.empty:
        st.error("Could not load site inventory")
        return

    _watershed_view(inventory_df)
