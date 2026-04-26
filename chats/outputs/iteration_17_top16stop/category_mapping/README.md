# Iteration 17 - Category Mapping Summary

## Overview
- **Iteration**: 17 (Top-16 Generic Stopwords)
- **Clusters**: 54 mapped to 16 categories
- **Silhouette Score**: 0.626
- **Total Conversations**: 24,119
- **Categorized**: 18,033 (74.8%)
- **Noise**: 6,086 (25.2%)

## Category Distribution

| # | Category | Conversations | % |
|---|---------|---------------|---|
| 14 | General/Gral | 4,860 | 27.0% |
| 7 | Productos de Crédito | 2,300 | 12.8% |
| 2 | Activación y Tokens | 2,221 | 12.3% |
| 1 | Atención a Clientes | 1,557 | 8.6% |
| 4 | Estados de Cuenta | 1,475 | 8.2% |
| 3 | Cargos No Reconocidos | 904 | 5.0% |
| 8 | Pagos y Comisiones | 778 | 4.3% |
| 12 | Fidelización/Membresía | 712 | 3.9% |
| 5 | Apple Pay/Billetera | 600 | 3.3% |
| 16 | Aclaraciones | 527 | 2.9% |
| 10 | Transferencias | 475 | 2.6% |
| 13 | Términos de Financiamiento | 455 | 2.5% |
| 9 | Retiros en Efectivo | 415 | 2.3% |
| 6 | Cancelación de Cuenta | 328 | 1.8% |
| 15 | Tarjeta Bloqueada | 299 | 1.7% |
| 11 | Depósitos | 127 | 0.7% |

## Category Definitions

| Category ID | Name | Description |
|-------------|------|-------------|
| 1 | Atención a Clientes | Customer service contacts, phone numbers, speak with advisor |
| 2 | Activación y Tokens | Card activation, token issues, virtual card, codi |
| 3 | Cargos No Reconocidos | Unrecognized charges, disputes, chargebacks |
| 4 | Estados de Cuenta | Account statements, CLABE, tracking keys, movements |
| 5 | Apple Pay/Billetera | Apple Pay, digital wallet, payment options |
| 6 | Cancelación de Cuenta | Account cancellation, closure requests |
| 7 | Productos de Crédito | Credit cards, personal loans, auto credit, limits |
| 8 | Pagos y Comisiones | Payments, fees, payment dates, airtime |
| 9 | Retiros en Efectivo | Cash withdrawals, ATM without card |
| 10 | Transferencias | Bank transfers, debt transfer |
| 11 | Depósitos | Cash deposits, OXXO deposits |
| 12 | Fidelización/Membresía | Loyalty programs, Hey Pro, Pal Norte, NIP |
| 13 | Términos de Financiamiento | Financing terms, months, down payment |
| 14 | General/Gral | Greetings, thanks, general questions |
| 15 | Tarjeta Bloqueada | Blocked card issues |
| 16 | Aclaraciones | Clarification requests, how-to questions |

## Files Generated

| File | Description |
|------|-------------|
| `mapping.csv` | Cluster-to-category mapping |
| `category_summary.csv` | Category counts summary |
| `first_turns_categorized.parquet` | Full dataset with category labels |
| `category_distribution.png` | Visualization charts |

## Top-Level Business Insights

1. **27% are General inquiries** - High-level questions not specific to a product
2. **25.8% are Card-related** (Activación+Tokens + Tarjeta Bloqueada) - Card issues are significant
3. **20.5% are Credit products** - Large volume of credit-related inquiries
4. **8.6% need human attention** - Customer service contacts
5. **5% are charge disputes** - Potential churn indicators