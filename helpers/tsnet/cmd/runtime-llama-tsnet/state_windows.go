//go:build windows

package main

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

func checkPrivate(path string, _ os.FileInfo, directory bool) error {
	tokenUser, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		return errors.New("invalid_state")
	}
	sd, err := windows.GetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.OWNER_SECURITY_INFORMATION)
	if err != nil {
		return errors.New("invalid_state")
	}
	owner, _, err := sd.Owner()
	if err != nil || !owner.Equals(tokenUser.User.Sid) {
		return errors.New("invalid_state")
	}
	flags := ""
	if directory {
		flags = "OICI"
	}
	private, err := windows.SecurityDescriptorFromString("D:P(A;" + flags + ";FA;;;" + owner.String() + ")")
	if err != nil {
		return errors.New("invalid_state")
	}
	dacl, _, err := private.DACL()
	if err != nil {
		return errors.New("invalid_state")
	}
	return windows.SetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION, nil, nil, dacl, nil)
}

func lockFile(file *os.File) error {
	return windows.LockFileEx(windows.Handle(file.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &windows.Overlapped{})
}

func unlockFile(file *os.File) error {
	return windows.UnlockFileEx(windows.Handle(file.Fd()), 0, 1, 0, &windows.Overlapped{})
}
