import awac.run_awac_cldvorl_ablation as awac
import dt.run_dt_cldvorl_ablation as dt
import edac.run_edac_cldvorl_ablation as edac
import inac.run_inac_cldvorl_ablation as inac
import iql.run_iql_cldvorl_ablation as iql
import sacn.run_sacn_cldvorl_ablation as sacn
import td3bc.run_td3bc_cldvorl_ablation as td3bc
import td7.run_td7_cldvorl_ablation as td7
import xql.run_xql_cldvorl_ablation as xql

import random

from UtilsRL.exp import parse_args

args = parse_args()

if args.baseline == 'awac':
    awac.main()
elif args.baseline == 'dt':
    dt.main()
elif args.baseline == 'edac':
    edac.main()
elif args.baseline == 'inac':
    inac.main()
elif args.baseline == 'iql':
    iql.main()
elif args.baseline == 'sacn':
    sacn.main()
elif args.baseline == 'td3bc':
    td3bc.main()
elif args.baseline == 'td7':
    td7.main()
elif args.baseline == 'xql':
    xql.main()
