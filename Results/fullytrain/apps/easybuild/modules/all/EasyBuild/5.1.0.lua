help([==[

Description
===========
EasyBuild is a software build and installation framework
 written in Python that allows you to install software in a structured,
 repeatable and robust way.


More information
================
 - Homepage: https://easybuilders.github.io/easybuild
]==])

whatis([==[Description: EasyBuild is a software build and installation framework
 written in Python that allows you to install software in a structured,
 repeatable and robust way.]==])
whatis([==[Homepage: https://easybuilders.github.io/easybuild]==])
whatis([==[URL: https://easybuilders.github.io/easybuild]==])

local root = "/apps/easybuild/software/EasyBuild/5.1.0"

conflict("EasyBuild")

prepend_path("CMAKE_PREFIX_PATH", root)
prepend_path("PATH", pathJoin(root, "bin"))
setenv("EBROOTEASYBUILD", root)
setenv("EBVERSIONEASYBUILD", "5.1.0")
setenv("EBDEVELEASYBUILD", pathJoin(root, "easybuild/EasyBuild-5.1.0-easybuild-devel"))

prepend_path("PYTHONPATH", pathJoin(root, "lib/python3.9/site-packages"))
setenv("EB_INSTALLPYTHON", "/usr/bin/python3")
-- Built with EasyBuild version 5.0.0
