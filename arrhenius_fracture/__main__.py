import sys

if len(sys.argv) > 1 and sys.argv[1] == "v12-production":
    from .v12_production_driver import main
    del sys.argv[1]
else:
    from .sharp_front import main

if __name__ == "__main__":
    main()
