# RoBERTa INT4 Multi-task Takeaways

- Main table uses only full seed-16 rows and only the pre-declared policies: fixed-small, MeZO default, and frozen analytical reference.
- Dense rows are retained in the appendix because a complete fixed-small/default/reference comparison is not available in one clean configuration.
- Fixed-small often fails; default is competitive in broad-window settings.
- Prefix INT4 provides the clearest default-failure/recovery evidence.
- The reference radius is not claimed to universally beat default.
