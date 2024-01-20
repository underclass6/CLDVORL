import awac.run_awac_withoutcl_ablation as awac
import iql.run_iql_withoutcl_ablation as iql
import td3bc.run_td3bc_withoutcl_ablation as td3bc

from UtilsRL.exp import parse_args

args = parse_args()

if args.baseline == 'awac':
    awac.main()

elif args.baseline == 'iql':
    iql.main()

elif args.baseline == 'td3bc':
    td3bc.main()

