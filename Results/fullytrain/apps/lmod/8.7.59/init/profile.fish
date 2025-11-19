#!/usr/bin/env fish
# -*- fish-shell-script -*-
########################################################################
#  This is the system wide source file for setting up
#  modules in Fish:
#
########################################################################

if test -z "$LMOD_ALLOW_ROOT_USE"
    set LMOD_ALLOW_ROOT_USE "yes"
end

if test $LMOD_ALLOW_ROOT_USE != yes
    if test (id -u) = 0
       exit 0
    end
end	

if test -z "$MODULEPATH_ROOT"

    if test -n "$USER"
        set -gx USER "$LOGNAME"  # make sure $USER is set
    end
    set -gx LMOD_sys (uname)

    set -gx MODULEPATH_ROOT "/apps/modulefiles"

    if test -z "$LMOD_MODULEPATH_INIT"
        if test -e /etc/lmod/.modulespath
            set LMOD_MODULEPATH_INIT "/etc/lmod/.modulespath"
        else
            set LMOD_MODULEPATH_INIT "/apps/lmod/lmod/init/.modulespath"
        end
    end
    
    if test -e "$LMOD_MODULEPATH_INIT" 
        for str in (cat "$LMOD_MODULEPATH_INIT" | sed 's/#.*$//')  # Allow end-of-line comments.
            for dir in (/usr/bin/ls -d "$str")
                set -gx MODULEPATH (/apps/lmod/lmod/libexec/addto --append MODULEPATH $dir)
            end
        end
    else
        set -xg MODULEPATH (/apps/lmod/lmod/libexec/addto --append MODULEPATH $MODULEPATH_ROOT/$LMOD_sys $MODULEPATH_ROOT/Core)
        set -xg MODULEPATH (/apps/lmod/lmod/libexec/addto --append MODULEPATH /apps/lmod/lmod/modulefiles/Core)
    end

    #################################################################
    # Prepend any directories in LMOD_SITE_MODULEPATH to $MODULEPATH
    #################################################################
    if test -n "$LMOD_SITE_MODULEPATH"
      set -gx MODULEPATH (/apps/lmod/lmod/libexec/addto MODULEPATH $LMOD_SITE_MODULEPATH)
    end

    set -xg FISH_ENV /apps/lmod/lmod/init/fish

    #
    # If MANPATH is empty, Lmod is adding a trailing ":" so that
    # the system MANPATH will be found
    if test -z "$MANPATH"
        set -xg MANPATH ":"
    end

    set -gx MANPATH (/apps/lmod/lmod/libexec/addto MANPATH /apps/lmod/lmod/share/man)
end
source  /apps/lmod/lmod/init/fish >/dev/null # Module Support

set fish_complete_path /apps/lmod/lmod/init/fish_tab_completion $fish_complete_path

