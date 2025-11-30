import streamlit as st

st.set_page_config(
    page_title="ODI Batting Performance – Debug Mode",
    page_icon="🏏",
    layout="wide",
)

st.title("🏏 ODI Batting Prediction – Debug Mode")

st.write("This page is here to debug why the deployed app was blank.")

try:
    st.subheader("1️⃣ Importing project_model.py...")
    import project_model
    st.success("✅ Successfully imported `project_model` module.")

    st.subheader("2️⃣ Calling project_model.get_trained_objects()...")
    try:
        trained = project_model.get_trained_objects()
        st.success("✅ Successfully called get_trained_objects().")

        st.write("Keys returned by get_trained_objects():")
        st.write(list(trained.keys()))

        st.info(
            "If you see keys like `model`, `feature_columns`, `unique_opponents`, etc., "
            "then the backend is OK and we can build the full dashboard next."
        )

    except Exception as e:
        st.error("❌ Error inside get_trained_objects()")
        st.exception(e)

except Exception as e:
    st.error("❌ Error importing `project_model.py`")
    st.exception(e)

st.markdown("---")
st.write(
    "Once this debug page shows everything green (no errors above), "
    "we'll switch back to the full dashboard UI."
)
